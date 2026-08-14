import lumi_paths as L
import os
import random
from typing import Any, Optional, Union
from functools import lru_cache

from omegaconf import OmegaConf
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA

# =============================================================================
# Target-variable preprocessing / normalisation
# =============================================================================
# TREFHT: Kelvin → Celsius then z-score-style with fixed (mean=4.5°C, std=21°C)
# pr:     kg/m²/s → mm/day then cube-root compression of the heavy positive tail
# PRECT:  m/s → mm/day then log1p compression of the heavy positive tail + z-score

# ── PRECT normalisation constants (log1p(mm/day) mean & std over hist+ssp370) ──
# Estimated 2026-06-12 with scripts/estimate_prect_norm.py on the clean
# re-downloaded realization LE2-1001.001, pooled hist (1850-2014) + ssp370
# (2015-2100), n=13.9M gridpoint-years. If the staging recipe or member set
# changes materially, re-run the script and update these.
PRECT_LOG_MEAN = 1.0727
PRECT_LOG_STD  = 0.5703

MIN_MAX_CONSTANTS = {"TREFHT": (-85.0, 60.0), "pr": (0.0, 6.0), "PRECT": (0.0, 50.0)}

PREPROCESS_FN = {
    "TREFHT": lambda x: x - 273.15,
    "pr": lambda x: x * 86400,
    # m/s → mm/day: ×1000 (m→mm) ×86400 (s→day) = 8.64e7
    "PRECT": lambda x: x * 8.64e7,
}
NORM_FN = {
    "TREFHT": lambda x: (x - 4.5) / 21.0,
    "pr": lambda x: np.cbrt(x),
    # log1p compresses the positive tail (no epsilon: log1p(0)=0, mm/day ≥ 0),
    # then z-score so the channel sits at ~unit variance to balance the MSE
    # against TREFHT's (x-4.5)/21.
    "PRECT": lambda x: (np.log1p(x) - PRECT_LOG_MEAN) / PRECT_LOG_STD,
}
DENORM_FN = {
    "TREFHT": lambda x: x * 21.0 + 4.5,
    "pr": lambda x: x**3,
    # returns mm/day
    "PRECT": lambda x: np.expm1(x * PRECT_LOG_STD + PRECT_LOG_MEAN),
}


def preprocess(ds: xr.DataArray) -> xr.DataArray:
    """Apply variable-specific unit conversion (Kelvin→Celsius, kg/m²/s→mm/day)."""
    return PREPROCESS_FN[ds.name](ds)


# =============================================================================
# Conditioning (CO2 / SUL) normalisation
# =============================================================================
# Reference scenarios used to derive the [-1, +1] percentile range.
# `_only_timefixed.nc` files contain scenario emissions only (no historical
# baseline), matching what config_data.yaml feeds to training/eval.
# ssp126 is intentionally excluded: it's the OOD test scenario.
EMISSIONS_PATHS = [
    f"{L.DATA}/emissions_hist_only_timefixed_bc.nc",
    f"{L.DATA}/emissions_ssp370_only_timefixed_bc.nc",
    f"{L.DATA}/emissions_aaer_only_timefixed_bc.nc",
    f"{L.DATA}/emissions_ghg_only_timefixed_bc.nc",
]


# -----------------------------------------------------------------------------
# Alternative normalisation functions kept for diagnostic scripts only
# (plot_hist.py, plot_difanos_cond_encoder.py). NOT used by the training path —
# the active normaliser is `normalize()` below, which uses
# `_get_emissions_minmax()`.
# -----------------------------------------------------------------------------

def scale_cumulative_linear(da: xr.DataArray):
    """Collapse to spatial mean per year, normalize to [-1, 1], broadcast back.
    Preserves temporal signal perfectly; every grid cell gets the same
    value per year (the global mean emission level for that year).
    Best for well-mixed gases like CO2."""
    spatial_dims = [d for d in da.dims if d not in ("year", "time")]
    ts = da.mean(dim=spatial_dims)  # [year]
    lo = float(ts.min(skipna=True))
    hi = float(ts.max(skipna=True))
    normed = (2.0 * (ts - lo) / max(hi - lo, 1e-30) - 1.0)
    return normed.broadcast_like(da).astype("float32")


def scale_emis_0_1_log10(da: xr.DataArray, low_pct=1.0, high_pct=99.0, floor=1e-30):
    # TOMCAT emissions: non-negative
    x = da.clip(min=0)
    # avoid log(0)
    x = xr.where(x > 0, x, floor)

    lx = np.log10(x)

    lo = lx.quantile(low_pct/100.0, skipna=True)
    hi = lx.quantile(high_pct/100.0, skipna=True)

    z = (lx - lo) / (hi - lo)
    return z.fillna(0).astype("float32")

def scale_emis_m1_p1_log10(da: xr.DataArray, low_pct=1.0, high_pct=99.99999999, floor=1e-30):
    z01 = scale_emis_0_1_log10(da, low_pct, high_pct, floor)
    return (2.0 * z01 - 1.0).astype("float32")


def scale_quantile_transform(da: xr.DataArray, n_quantiles=1000, floor=1e-30):
    """sklearn QuantileTransformer: rank-based normalization to [-1, 1]."""
    shape = da.shape
    vals = da.values.copy()

    real_mask = vals

    if real_mask.sum() == 0:
        return xr.DataArray(
            np.full(shape, -1.0, dtype=np.float32),
            dims=da.dims, coords=da.coords,
        )

    real_vals = vals[real_mask].reshape(-1, 1)

    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(real_vals)),
        output_distribution='uniform',
        random_state=42,
    )
    qt.fit(real_vals)

    transformed = np.full(shape, -1.0, dtype=np.float32)
    transformed[real_mask] = (
        2.0 * qt.transform(vals[real_mask].reshape(-1, 1)).ravel() - 1.0
    )
    del vals, real_vals

    return xr.DataArray(
        transformed, dims=da.dims, coords=da.coords,
    ).astype("float32")

# ---------------------------------------------------------------------------
# PCA denoising helpers
# ---------------------------------------------------------------------------

def smooth_cond_spatial(arr, sigmas, method="gaussian", var_names=None):
    """Per-channel spatial smoothing of cond fields, shape (n_vars, T, H, W).

    Longitude (last axis) is periodic → wrap padding; latitude → reflect.
      method="gaussian": separable gaussian_filter1d — blurs ALL scales, so it
        also erases the regional aerosol fingerprint along with the line noise.
      method="median": k×k median filter, k=2*round(sigma)+1 (sigma=1→3×3,
        2→5×5) — removes thin line artefacts (shipping lanes, flight paths)
        while preserving broad continental structure.
    Returns a new array; channels with sigma<=0 are left unchanged.
    """
    arr = np.asarray(arr).copy()
    method = str(method).lower()
    for v_idx, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        ch = arr[v_idx]                                   # (T, H, W)
        name = var_names[v_idx] if var_names is not None else f"ch{v_idx}"
        if method == "median":
            from scipy.ndimage import median_filter
            k = 2 * int(round(sigma)) + 1
            pad = k // 2
            ch = np.pad(ch, ((0, 0), (0, 0), (pad, pad)), mode="wrap")     # lon periodic
            ch = np.pad(ch, ((0, 0), (pad, pad), (0, 0)), mode="reflect")  # lat
            ch = median_filter(ch, size=(1, k, k), mode="nearest")
            ch = ch[:, pad:-pad, pad:-pad]
            print(f"[COND] spatial median filter {k}x{k} applied to {name}")
        else:
            from scipy.ndimage import gaussian_filter1d
            ch = gaussian_filter1d(ch, sigma=sigma, axis=-1, mode="wrap")
            ch = gaussian_filter1d(ch, sigma=sigma, axis=-2, mode="reflect")
            print(f"[COND] spatial gaussian smoothing sigma={sigma} applied to {name}")
        arr[v_idx] = ch
    return arr


def fit_pca_denoise(
    data: np.ndarray,
    n_components: int,
    var_name: str = "",
) -> tuple[np.ndarray, PCA]:
    T, H, W = data.shape
    flat = data.reshape(T, H * W).astype(np.float64)  # PCA needs float64

    # Guard: constant field (e.g. SUL=0 in GHG-only scenario) has zero
    # variance — sklearn PCA divides by singular values → NaN in components.
    # Return the channel unchanged and fit a 1-component PCA just to produce
    # a valid object that apply_pca_denoise can use without crashing.
    if flat.std() < 1e-8:
        pca = PCA(n_components=1, whiten=False).fit(flat)
        return data.astype(np.float32), pca

    n_components = min(n_components, T, H * W)
    pca = PCA(n_components=n_components, whiten=False)
    scores = pca.fit_transform(flat)           # (T, n_components)
    del flat                                   # free before inverse_transform
    recon = pca.inverse_transform(scores)      # (T, H*W)
    del scores

    result = recon.reshape(T, H, W).astype(np.float32)
    del recon
    return result, pca


def apply_pca_denoise(data: np.ndarray, pca: PCA) -> np.ndarray:
    """Project new data through an already-fitted PCA and reconstruct."""
    T, H, W = data.shape
    flat = data.reshape(T, H * W).astype(np.float64)
    scores = pca.transform(flat)
    del flat
    recon = pca.inverse_transform(scores)
    del scores
    result = recon.reshape(T, H, W).astype(np.float32)
    del recon
    return result


def pca_denoise_dataset(
    tensor: torch.Tensor,           # (n_vars, T, H, W)
    n_components: Union[int, list[int]],
    var_names: Optional[list] = None,
    pca_objects: Optional[list] = None,
) -> tuple[torch.Tensor, list[PCA]]:
    """Apply PCA denoising to every variable channel of a tensor.

    If ``pca_objects`` is ``None`` the PCA is *fitted* on this data (use for
    the first realization / training time).  Otherwise the supplied fitted
    PCAs are *applied* without refitting (use for subsequent realizations and
    at generation time).

    Args:
        tensor:       Shape ``(n_vars, T, H, W)``.
        n_components: Components to retain when fitting.  Pass a single ``int``
                      to use the same count for every channel, or a ``list``
                      of ints (one per channel) for per-channel control.
                      e.g. ``[10, 40]`` keeps 10 EOFs for CO2 and 40 for SO2.
                      Only used when ``pca_objects`` is ``None``.
        var_names:    Optional list of variable names for diagnostic prints.
        pca_objects:  Pre-fitted PCA objects or ``None``.

    Returns:
        denoised_tensor: Same shape as input.
        pca_objects:     List of fitted :class:`PCA` objects (one per var).
    """
    n_vars = tensor.shape[0]
    var_names = var_names or [str(i) for i in range(n_vars)]

    # Normalise to a per-channel list
    if isinstance(n_components, int):
        n_components_list = [n_components] * n_vars
    else:
        if len(n_components) != n_vars:
            raise ValueError(
                f"n_components has {len(n_components)} entries but tensor has "
                f"{n_vars} channels.  Supply one value per channel or a single int."
            )
        n_components_list = list(n_components)

    np_data = tensor.numpy()
    denoised = np.empty_like(np_data)
    fitted_pcas: list[PCA] = []

    for v in range(n_vars):
        channel = np_data[v]  # (T, H, W) — view, no copy
        if pca_objects is None:
            recon, pca = fit_pca_denoise(channel, n_components_list[v], var_names[v])
            fitted_pcas.append(pca)
        else:
            recon = apply_pca_denoise(channel, pca_objects[v])
            fitted_pcas.append(pca_objects[v])
        denoised[v] = recon
        del recon

    del np_data
    result = torch.from_numpy(denoised)
    del denoised
    return result, fitted_pcas


# ---------------------------------------------------------------------------


# =============================================================================
# Active cond-normalisation path (used by the training pipeline)
# =============================================================================

# Per-channel clip percentiles (lo, hi) for the normalize() linear map.
# CO2 and SUL are both heavy-tailed (rare hotspots), so the top anchor sets how
# much the populated, signal-carrying gridpoints get compressed toward -1:
#   - CO2 → (1, 99): wide top so high-emission futures (ssp370/ssp126) don't
#     saturate at +1 (the af8bfcf fix; see cond_normalization_diag).
#   - SUL → (5, 95): tighter top restores ~10× more usable aerosol contrast.
#     Under (1, 99) the inhabited SUL field flattens onto the -1 floor (nonzero
#     p90 → -0.94 vs -0.04 at 5-95), starving the aerosol-only (aaer) signal and
#     making its response spiky/unstable. SUL gains nothing from the wider range.
# Splitting the percentile per channel resolves that CO2-vs-SUL conflict.
#   - BC → (5, 95): heavy-tailed combustion emissions, same hotspot geography as
#     SO2. SUL's 5-95 reasoning applies identically; do NOT use CO2's (1, 99).
_CLIP_PCTL = {"CO2": (1, 99), "SUL": (5, 95), "SO2": (5, 95), "sul": (5, 95),
              "BC": (5, 95)}

# ── BC clip mode ─────────────────────────────────────────────────────────────
# "v1"        — percentiles over ALL gridpoints, zeros included (5, 95). Under
#               this the populated BC field is semi-flattened: populated p50
#               normalizes to -0.996 and the temporal global-mean swing is ~3×
#               smaller than SUL's (BC is even more hotspot-concentrated than
#               SO2, so SUL's percentile choice doesn't transfer).
# "populated" — hi anchor from the POSITIVE values only, (5, 90). The populated
#               p90 (~1.36e-8, ≈½ the v1 anchor) trades ~11% of populated
#               hotspot cores clipping at +1 (SUL saturates 6.3%) for MEASURED
#               temporal-contrast gains of +16% (hist gmean swing 0.159→0.184)
#               and +59% (ssp370, 0.054→0.086). NOTE: no linear anchor can
#               reach SUL-like contrast (0.47) — an anchor low enough to lift the
#               mid-range clips the fastest-growing hotspot cells and deletes
#               their temporal signal (empirical scan 2026-07-03: best possible
#               hist swing ≈0.22 at 15% clipped). A real fix needs a nonlinear
#               transform, and log-scaling cond previously FAILED (see
#               feedback_log_normalization) — do not retry it casually.
# Changing the mode changes the meaning of the BC channel → fresh training
# required. The (lo, hi) actually used in training is persisted in the
# checkpoint under "COND_NORM" and re-injected at eval via
# set_minmax_override(), so old checkpoints keep evaluating with v1 no matter
# what this module's default is. Select via `bc_clip_mode` in the data config.
_BC_CLIP_MODE = "v1"
_BC_POPULATED_PCTL = (5, 90)
_MINMAX_OVERRIDE = None


def set_bc_clip_mode(mode: str) -> None:
    """Select the BC clip mode ("v1" | "populated") BEFORE datasets are built."""
    global _BC_CLIP_MODE
    if mode not in ("v1", "populated"):
        raise ValueError(f"unknown bc_clip_mode {mode!r} (expected 'v1' or 'populated')")
    _BC_CLIP_MODE = mode
    _get_emissions_minmax.cache_clear()


def set_minmax_override(minmax: "dict | None") -> None:
    """Inject checkpoint-persisted per-channel (lo, hi) clip ranges.

    Overrides the recomputed percentiles entirely so eval/resume normalizes
    cond exactly as the loaded checkpoint's training run did. Pass None to
    CLEAR a previously set override (a process loading a second checkpoint
    without COND_NORM must not inherit the first one's ranges).
    """
    global _MINMAX_OVERRIDE
    if minmax is None:
        _MINMAX_OVERRIDE = None
    else:
        _MINMAX_OVERRIDE = {k: (float(v[0]), float(v[1]))
                            for k, v in minmax.items()}
    _get_emissions_minmax.cache_clear()


def get_active_minmax() -> dict:
    """The per-channel (lo, hi) currently in effect — persisted to checkpoints."""
    return dict(_get_emissions_minmax())


@lru_cache(maxsize=1)
def _get_emissions_minmax():
    """Compute the per-channel clip percentile range across reference scenarios.

    Cached: opens the EMISSIONS_PATHS NetCDFs once per process. The returned
    (lo, hi) per variable defines the linear mapping in `normalize()`:
    lo → -1, hi → +1, values outside are clipped to [-1, +1]. Percentiles are
    per-channel via _CLIP_PCTL (CO2 1-99, SUL 5-95); BC honours _BC_CLIP_MODE.
    A checkpoint-injected override (set_minmax_override) short-circuits the
    computation entirely.
    """
    if _MINMAX_OVERRIDE is not None:
        return _MINMAX_OVERRIDE
    all_vals = {}  # var -> list of flat arrays
    for path in EMISSIONS_PATHS:
        ds_emis = xr.open_dataset(path)
        for var in ["CO2", "SO2", "SUL", "sul", "BC"]:
            if var not in ds_emis.data_vars:
                continue
            all_vals.setdefault(var, []).append(ds_emis[var].values.flatten())
        ds_emis.close()
    combined = {}
    for var, arrays in all_vals.items():
        flat = np.concatenate(arrays)
        if var == "BC" and _BC_CLIP_MODE == "populated":
            flat = flat[flat > 0]
            if flat.size == 0:
                raise ValueError(
                    "bc_clip_mode=populated: no positive BC values found in "
                    "EMISSIONS_PATHS — the BC field is all zeros/NaN "
                    "(corrupt or mis-staged cond files?)."
                )
            plo, phi = _BC_POPULATED_PCTL
        else:
            plo, phi = _CLIP_PCTL.get(var, (1, 99))
        combined[var] = (float(np.percentile(flat, plo)), float(np.percentile(flat, phi)))
    return combined


def normalize(ds: xr.DataArray) -> xr.DataArray:
    """Normalise a DataArray for model input.

    CO2 and SUL: min-max scaling derived from the 1st–99th percentile of the
    reference scenarios in EMISSIONS_PATHS (hist + ssp370 + aaer + ghg; ssp126
    excluded as OOD test). Percentile lo → -1, hi → +1; out-of-range values
    are clipped. This preserves spatial structure while preventing extreme
    hotspot gridpoints from dominating the range.

    Other variables (e.g. TREFHT, pr): use the fixed lambdas in NORM_FN.
    """
    if ds.name in ["CO2", "SUL", "BC"]:
        minmax = _get_emissions_minmax()
        min_val, max_val = minmax[ds.name]

        range_val = max_val - min_val
        if range_val == 0:
            return xr.zeros_like(ds).clip(-1, 1).fillna(-1)
        mean_val = (min_val + max_val) / 2
        norm = (ds - mean_val) / (range_val / 2)
        return norm.clip(-1, 1).fillna(-1)

    return NORM_FN[ds.name](ds).fillna(0)

def denorm(ds: xr.DataArray) -> xr.DataArray:
    norm = DENORM_FN[ds.name](ds)

    min_val, max_val = MIN_MAX_CONSTANTS[ds.name]
    # norm = min_max_denorm(norm, min_val, max_val)
    return norm


class ClimateDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        realizations: list[str],
        data_dir: str,
        target_vars: list[str],
        cond_file: str,
        cond_vars: list[str],
        # ── PCA denoising ────────────────────────────────────────────────────
        # Pass None to disable, a single int to use the same count for every
        # channel, or a list of ints (one per variable) for per-channel control.
        # e.g.  n_components_cond=[10, 40]  → 10 EOFs for CO2, 40 for SO2
        n_components_target: Optional[Union[int, list[int]]] = None,
        n_components_cond:   Optional[Union[int, list[int]]] = None,
        # ── previous-state conditioning (AR-1 style), OPT-IN ──────────────────
        # Append the PREVIOUS timestep's first target variable as an extra
        # conditioning channel, so the model can carry year-to-year memory.
        # Measured motivation: CESM2's global-mean temperature has lag-1
        # autocorrelation +0.435 while the emulator's is -0.028, because f=1
        # makes every year an independent sample (scripts/diag_interannual.py).
        # CESM2's lag-2 is only +0.077 — far below the 0.189 an AR(1) with
        # phi=0.435 would give — so ONE previous step carries essentially all
        # of the memory and a single extra channel is the right size of fix.
        #
        # DEFAULT OFF: with it off the cond tensor is byte-identical to before,
        # so existing checkpoints and configs are unaffected.
        prev_target_channel: bool = False,
        # Per-channel spatial gaussian σ (in gridpoints) applied to normalised
        # cond fields before PCA.  None or 0 disables smoothing for that channel.
        # Used to suppress fine-scale inventory artefacts (shipping lanes, flight
        # paths) in SUL that would otherwise leak into the predicted TREFHT.
        cond_smooth_sigma: Optional[Union[float, list[float]]] = None,
        # Denoiser for cond smoothing: "gaussian" (blurs everything — kills the
        # regional aerosol fingerprint along with the lines) or "median" (removes
        # thin line artefacts while preserving broad continental structure).
        # For median, kernel size = 2*round(sigma)+1 (sigma=1→3×3, sigma=2→5×5).
        cond_smooth_method: str = "gaussian",
        # ── Set True to skip loading target climate files (diagnostics only) ─
        cond_only: bool = False,
        # ── Name of the time dimension in the NetCDF files ───────────────────
        # "year" for older CESM2-LE files, "time" for AAER/GHG/ssp files
        time_dim: str = "year",
        # ── Pre-computed baseline for experiments without historical coverage ──
        # Pass the .climatology tensor from a historical ClimateDataset here
        # so that SSP370 (2015-2100) uses the same 1850-1900 baseline as hist.
        # Shape must be (1, n_vars, 1, H, W).  If None, baseline is computed
        # from the loaded data (requires data to cover 1850-1900).
        external_climatology: Optional[torch.Tensor] = None,
    ):
        self.seq_len = seq_len
        self.prev_target_channel = prev_target_channel
        self.realizations = realizations
        # data_dir may be a single tree (one mfdataset holding ALL target vars,
        # legacy) or a LIST of one tree per target var (e.g. TREFHT and PRECT
        # staged in separate dirs). Normalise to a list; the first element is
        # the "primary" dir used for diagnostics / relative cond_file resolution.
        if isinstance(data_dir, (list, tuple)) or (
            not isinstance(data_dir, str) and hasattr(data_dir, "__iter__")
        ):
            self._data_dirs = [str(d) for d in data_dir]
        else:
            self._data_dirs = [str(data_dir)]
        self.data_dir = self._data_dirs[0]
        self.cond_only = cond_only
        self.time_dim = time_dim

        # Necessary to convert vars into a Python list
        self.vars = OmegaConf.to_object(target_vars) if not isinstance(target_vars, list) else target_vars
        self.cond_vars = OmegaConf.to_object(cond_vars) if not isinstance(cond_vars, list) else cond_vars

        # Normalise n_components_* to a list (one entry per channel) or None.
        # This is done once here so load_data always receives a consistent type.
        self.n_components_target = self._norm_n_components(
            n_components_target, len(self.vars), "target"
        )
        self.n_components_cond = self._norm_n_components(
            n_components_cond, len(self.cond_vars), "cond"
        )

        # Normalise cond_smooth_sigma to a per-channel list (or None).
        if cond_smooth_sigma is None:
            self.cond_smooth_sigma = None
        else:
            if isinstance(cond_smooth_sigma, (int, float)):
                sigmas = [float(cond_smooth_sigma)] * len(self.cond_vars)
            else:
                sigmas = [float(s) for s in cond_smooth_sigma]
            if len(sigmas) != len(self.cond_vars):
                raise ValueError(
                    f"cond_smooth_sigma length {len(sigmas)} != cond_vars "
                    f"length {len(self.cond_vars)}"
                )
            # None if all zero — avoid the import + per-channel loop entirely
            self.cond_smooth_sigma = sigmas if any(s > 0 for s in sigmas) else None
        self.cond_smooth_method = str(cond_smooth_method).lower()

        # Fitted PCA objects – populated on first load_data call, then reused
        self._pca_target: Optional[list[PCA]] = None
        self._pca_cond: Optional[list[PCA]] = None

        # Store one dataset (out of memory) as an xarray dataset for metadata
        # Store a different dataset as a torch tensor for speed
        self.xr_data: Optional[xr.Dataset] = None
        self.tensor_data: Optional[torch.Tensor] = None
        self.lats = None
        self.cond_file = cond_file

        # Pre-industrial climatological baseline (1850–1900 mean) in normalised
        # model space.  Shape: (1, n_vars, 1, H, W) — broadcast-ready.
        # Populated on the first load_data() call; reused for all subsequent
        # realizations so the baseline is always consistent across training.
        # For experiments that don't cover 1850-1900 (e.g. SSP370), pass a
        # pre-computed baseline tensor via external_climatology instead.
        self.climatology: Optional[torch.Tensor] = external_climatology

        # Load an example realization right off the bat
        self.load_data(self.realizations[0])

    @staticmethod
    def _norm_n_components(
        value: Optional[Union[int, list[int]]],
        n_vars: int,
        label: str,
    ) -> Optional[list[int]]:
        """Normalise a PCA n_components spec to a per-channel list or None.

        Accepts:
          None          → PCA disabled, returns None
          int           → same count for every channel
          list[int]     → must match n_vars; returned as-is
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return [int(value)] * n_vars
        lst = list(value)          # handles ListConfig from OmegaConf
        lst = [int(v) for v in lst]
        if len(lst) != n_vars:
            raise ValueError(
                f"n_components_{label} has {len(lst)} entries but there are "
                f"{n_vars} {label} variable(s).  "
                f"Supply one value per channel or a single int."
            )
        return lst

    def estimate_num_batches(self, batch_size: int) -> int:
        """Estimates the number of batches in the dataset."""
        return len(self) * len(self.realizations) // batch_size

    def _select_years(self) -> set:
        """Years to load from each realization (targets AND conditioning).

        TRAINING SUBSAMPLES: every 5th historical year and every other future
        year, ~76 of the 251 available, to cut I/O and epoch time. The chunk
        files on disk hold every year — this decimation is applied at load time.

        Override in a subclass to change coverage; see EvalClimateDataset, which
        takes every year for evaluation/generation. Do NOT widen this default —
        it changes what every trained model was fitted on.
        """
        hist_years = list(range(1850, 2015, 5))    # every 5th year
        future_years = list(range(2015, 2101, 2))  # every other year
        return set(hist_years + future_years)

    def load_data(self, realization: str):
        """Loads the data from the specified paths and returns it as an xarray Dataset.

        When ``self.cond_only`` is True the target climate realization files
        are skipped entirely, which is useful for diagnostics and tools that
        only need the conditioning data.
        """
        selected_years = self._select_years()

        # ── Target climate data (skipped in cond_only mode) ──────────────────
        if not self.cond_only:
            # Close previous dataset to free file handles and memory
            if getattr(self, "xr_data", None) is not None:
                self.xr_data.close()
                self.xr_data = None
            if hasattr(self, "tensor_data"):
                del self.tensor_data
            self.tensor_data = None

            if len(self._data_dirs) == 1:
                # Legacy: one tree holds all target vars.
                realization_dir = os.path.join(self._data_dirs[0], realization, "*.nc")
                # Open lazily; only materialise when we call convert_xarray_to_tensor
                dataset = xr.open_mfdataset(
                    realization_dir, combine="by_coords", chunks={self.time_dim: 50}
                ).sortby(self.time_dim)
            else:
                # One tree per target var (e.g. TREFHT, PRECT staged separately),
                # paired by IDENTICAL realization dir name. Open each, select its
                # var, and merge on shared coords (inner join on the time axis).
                if len(self._data_dirs) != len(self.vars):
                    raise ValueError(
                        f"data_dir has {len(self._data_dirs)} trees but there are "
                        f"{len(self.vars)} target_vars {self.vars} — supply one tree "
                        f"per target var (paired by realization name)."
                    )
                per_var = []
                for vname, ddir in zip(self.vars, self._data_dirs):
                    d = xr.open_mfdataset(
                        os.path.join(ddir, realization, "*.nc"),
                        combine="by_coords", chunks={self.time_dim: 50},
                    ).sortby(self.time_dim)
                    per_var.append(d[[vname]])
                dataset = xr.merge(per_var, join="inner")
                # Inner join silently DROPS years absent from any tree (e.g. a
                # partially-staged PRECT member) — that would quietly shrink the
                # training set. Demand identical time axes across the trees.
                n_merged = dataset.sizes[self.time_dim]
                for vname, d in zip(self.vars, per_var):
                    n_var = d.sizes[self.time_dim]
                    if n_var != n_merged:
                        raise ValueError(
                            f"{realization}: time axis mismatch across target-var "
                            f"trees — {vname} has {n_var} steps but the inner-join "
                            f"intersection is {n_merged}. A tree is partially "
                            f"staged; re-run run_prepare_data.sh for it."
                        )
            self.lats = dataset.lat
            # Pin channel order to target_vars (ch0=TREFHT, ch1=PRECT, …) so the
            # stacked tensor / denoiser never silently transposes the channels.
            dataset = dataset[self.vars]
            self.xr_data = dataset.map(preprocess).map(normalize)
            # Select target years robustly: handles both integer-year coords
            # (CESM2-LE "year" dim) and datetime/cftime coords (AAER/GHG/SSP files).
            # Also safe when a scenario doesn't cover the full 1850-2100 range.
            coord_vals = self.xr_data[self.time_dim].values
            if hasattr(coord_vals[0], 'year'):
                # cftime or datetime64 objects: extract integer year and isel by mask
                year_ints = [int(str(v)[:4]) for v in coord_vals]
                mask = [y in selected_years for y in year_ints]
                self.xr_data = self.xr_data.isel({self.time_dim: mask})
            else:
                # Integer year coordinate — intersect with what's in the file
                available = set(int(v) for v in coord_vals)
                valid = sorted(selected_years & available)
                self.xr_data = self.xr_data.sel({self.time_dim: valid})
            # Store year values before converting so we can close xr_data early
            self._time_values = self.xr_data[self.time_dim].values.astype(int)
            self.tensor_data = self.convert_xarray_to_tensor(self.xr_data)
            # Trigger compute and release the dask graph immediately
            self.tensor_data = self.tensor_data.contiguous()

            if self.n_components_target is not None:
                self.tensor_data, self._pca_target = pca_denoise_dataset(
                    self.tensor_data,
                    n_components=self.n_components_target,
                    var_names=self.vars,
                    pca_objects=self._pca_target,
                )

            # ── Climatological baseline (computed once, reused for all realizations)
            # We only compute it on the first load_data() call (self.climatology is
            # None) so every realization uses the same baseline.  If the dataset
            # does not cover 1850-1900 a warning is printed and we fall back to
            # None (the trainer will then use per-batch mean as before).
            if self.climatology is None:
                try:
                    self.climatology = self.get_baseline_mean(
                        baseline_start=1850, baseline_end=1900
                    )
                except RuntimeError as exc:
                    print(f"[DATASET] Could not compute climatology: {exc}")
                    print("[DATASET] Anomaly loss will fall back to per-batch mean.")
                    self.climatology = None

        # ── Conditioning data (always loaded) ────────────────────────────────
        if getattr(self, "dataset_cond", None) is not None:
            self.dataset_cond.close()
        if hasattr(self, "tensor_data_cond"):
            del self.tensor_data_cond
        self.tensor_data_cond = None

        cond_file = (
            self.cond_file
            if os.path.isabs(self.cond_file)
            else os.path.join(self.data_dir, self.cond_file)
        )
        # Open conditioning file lazily too
        raw_cond = xr.open_dataset(cond_file)
        # Cond files may use "year" as time coord while data uses self.time_dim ("time")
        if self.time_dim not in raw_cond.dims and "year" in raw_cond.dims:
            raw_cond = raw_cond.rename({"year": self.time_dim})
        raw_cond = raw_cond.chunk({self.time_dim: -1})
        raw_cond = raw_cond[self.cond_vars].map(normalize)#.sel({self.time_dim: selected_years})

        coord_vals = raw_cond[self.time_dim].values
        if hasattr(coord_vals[0], 'year'):
            # cftime or datetime64 objects: extract integer year and isel by mask
            year_ints = [int(str(v)[:4]) for v in coord_vals]
            mask = [y in selected_years for y in year_ints]
            raw_cond = raw_cond.isel({self.time_dim: mask})
        else:
            # Integer year coordinate — intersect with what's in the file
            available = set(int(v) for v in coord_vals)
            valid = sorted(selected_years & available)
            raw_cond = raw_cond.sel({self.time_dim: valid})

        # Materialise into a float32 tensor and immediately close the dataset
        self.tensor_data_cond = self.convert_xarray_to_tensor(raw_cond).contiguous()
        # Keep a lightweight (no-data) reference for coordinate lookups
        _ds_cond = xr.open_dataset(cond_file)
        if self.time_dim not in _ds_cond.dims and "year" in _ds_cond.dims:
            _ds_cond = _ds_cond.rename({"year": self.time_dim})
        self.dataset_cond = _ds_cond[self.cond_vars]#.sel({self.time_dim: selected_years})
        raw_cond.close()
        del raw_cond

        if self.prev_target_channel:
            self._append_prev_target_channel()

        # ── Spatial smoothing on conditioning (before PCA) ───────────────────
        # Applied per-channel via cond_smooth_sigma list. Removes line features
        # from gridded inventories (e.g. SO2 shipping lanes, flight paths) that
        # would otherwise teach the model non-physical per-pixel correlations.
        # Longitude axis uses wrap padding (periodic); latitude uses reflect.
        # method="gaussian" blurs everything (also erases the regional aerosol
        # fingerprint); "median" removes thin lines but keeps broad structure.
        if self.cond_smooth_sigma is not None:
            arr = smooth_cond_spatial(
                self.tensor_data_cond.numpy(),          # (n_vars, T, H, W)
                self.cond_smooth_sigma, self.cond_smooth_method, self.cond_vars)
            self.tensor_data_cond = torch.from_numpy(arr).contiguous()

        # ── PCA denoising on conditioning ────────────────────────────────────
        if self.n_components_cond is not None:
            self.tensor_data_cond, self._pca_cond = pca_denoise_dataset(
                self.tensor_data_cond,
                n_components=self.n_components_cond,
                var_names=self.cond_vars,
                pca_objects=self._pca_cond,     # None on first call → fits
            )

        # Save diagnostic spatial plots (only on first load)
        diag_dir = os.path.join(self.data_dir, "diagnostics")
        if not os.path.isdir(diag_dir):
            os.makedirs(diag_dir, exist_ok=True)
            self._save_cond_diagnostics(diag_dir)

    def _save_cond_diagnostics(self, diag_dir: str):
        """Save spatial maps and time series of conditioning data.

        When PCA denoising is enabled the plots show the PCA-filtered tensor
        (i.e. exactly what the model receives).  When PCA is disabled they
        fall back to the raw-normalised xarray values — the two are identical
        in that case, so the plots are always consistent with model input.
        """
        all_years = self.dataset_cond[self.time_dim].values
        candidate_years = [all_years[0], 2015, 2050, all_years[-1]]
        years_to_show   = [y for y in candidate_years if y in all_years]
        year_indices    = [int(np.where(all_years == y)[0][0]) for y in years_to_show]

        pca_active_cond = self._pca_cond is not None

        # tensor_data_cond shape: (n_vars, T, H, W) — already PCA-filtered if enabled
        cond_np = self.tensor_data_cond.numpy()   # (n_vars, T, H, W)

        # ── spatial maps ─────────────────────────────────────────────────────
        for v_idx, var in enumerate(self.cond_vars):
            channel = cond_np[v_idx]              # (T, H, W)
            n_years = len(years_to_show)

            fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 4))
            if n_years == 1:
                axes = [axes]

            for col, (yr, t_idx) in enumerate(zip(years_to_show, year_indices)):
                ax = axes[col]
                data = channel[t_idx]             # (H, W)
                im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                               vmin=-1, vmax=1, origin='lower')
                ax.set_title(
                    f"year={yr}\nmin={data.min():.3f} max={data.max():.3f}\n"
                    f"mean={data.mean():.3f} std={data.std():.3f}",
                    fontsize=9,
                )
                plt.colorbar(im, ax=ax, shrink=0.8)

            pca_label = (
                f" [PCA {self._pca_cond[v_idx].n_components_} comps, "
                f"{self._pca_cond[v_idx].explained_variance_ratio_.sum()*100:.1f}% var]"
                if pca_active_cond else ""
            )
            fig.suptitle(
                f"{var} — cond_map seen by model{pca_label}",
                fontsize=13,
            )
            plt.tight_layout()
            save_path = os.path.join(diag_dir, f"cond_normalized_{var}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        # ── spatial-mean time series ──────────────────────────────────────────
        fig, axes = plt.subplots(len(self.cond_vars), 1,
                                 figsize=(12, 4 * len(self.cond_vars)))
        if len(self.cond_vars) == 1:
            axes = [axes]

        for v_idx, var in enumerate(self.cond_vars):
            channel = cond_np[v_idx]              # (T, H, W)
            ts      = channel.mean(axis=(1, 2))   # (T,)  spatial mean

            ax = axes[v_idx]
            ax.plot(all_years, ts, 'b-', linewidth=2)

            pca_label = (
                f" [PCA {self._pca_cond[v_idx].n_components_} comps]"
                if pca_active_cond else ""
            )
            ax.set_title(
                f"{var} — spatial mean of cond_map seen by model{pca_label}\n"
                f"range: [{ts.min():.3f}, {ts.max():.3f}]",
                fontsize=12,
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Normalised value")
            ax.set_ylim(-1.15, 1.15)
            ax.axhline(-1, color='gray', ls='--', alpha=0.4)
            ax.axhline(1,  color='gray', ls='--', alpha=0.4)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(diag_dir, "cond_timeseries.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # ── PCA scree + before/after maps ────────────────────────────────────
        self._save_pca_diagnostics(diag_dir)

    def _save_pca_diagnostics(self, diag_dir: str):
        """Save scree plots and before/after spatial maps for PCA denoising.

        The "raw" panel is derived fresh from the normalised xarray dataset so
        it truly reflects the pre-PCA signal, while the "PCA filtered" panel
        comes from the tensor that the model actually receives.
        """
        pca_sets = [
            (
                "target",
                self._pca_target,
                self.vars,
                self.tensor_data,                              # already PCA-filtered
                self.convert_xarray_to_tensor(self.xr_data),  # raw normalised
            ),
            (
                "cond",
                self._pca_cond,
                self.cond_vars,
                self.tensor_data_cond,                                   # already PCA-filtered
                self.convert_xarray_to_tensor(self.dataset_cond),        # raw normalised
            ),
        ]

        for tag, pca_list, var_names, filtered_tensor, raw_tensor in pca_sets:
            if pca_list is None:
                continue  # PCA not enabled for this set

            for v_idx, (pca, vname) in enumerate(zip(pca_list, var_names)):
                # ── scree plot ───────────────────────────────────────────────
                cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(np.arange(1, len(cumvar) + 1), cumvar, 'o-', ms=4)
                ax.axhline(90, color='orange', ls='--', label='90 %')
                ax.axhline(95, color='red',    ls='--', label='95 %')
                ax.set_xlabel("Number of components")
                ax.set_ylabel("Cumulative variance explained (%)")
                ax.set_title(f"PCA scree — {tag}/{vname}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                scree_path = os.path.join(diag_dir, f"pca_scree_{tag}_{vname}.png")
                plt.savefig(scree_path, dpi=120, bbox_inches='tight')
                plt.close()

                # ── before / after spatial map ────────────────────────────────
                # Use the middle time-step for a representative snapshot
                mid_t = raw_tensor.shape[1] // 2

                raw_map      = raw_tensor[v_idx, mid_t].numpy()       # (H, W)  pre-PCA
                filtered_map = filtered_tensor[v_idx, mid_t].numpy()  # (H, W)  post-PCA
                resid_map    = raw_map - filtered_map                  # (H, W)  removed noise

                # Shared colour scale anchored on the raw field range
                vmin, vmax = raw_map.min(), raw_map.max()
                # Residual uses its own symmetric scale so small values show up
                rvmax = np.abs(resid_map).max()

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, data, title, vm0, vm1 in zip(
                    axes,
                    [raw_map,        filtered_map,                          resid_map],
                    ["Raw (normalised)", f"PCA filtered ({pca.n_components_} comps)",  "Noise removed (raw − filtered)"],
                    [vmin,           vmin,                                  -rvmax],
                    [vmax,           vmax,                                   rvmax],
                ):
                    im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                                   vmin=vm0, vmax=vm1, origin='lower')
                    ax.set_title(
                        f"{title}\nmin={data.min():.3f}  max={data.max():.3f}",
                        fontsize=9,
                    )
                    plt.colorbar(im, ax=ax, shrink=0.8)

                fig.suptitle(
                    f"PCA denoising — {tag}/{vname}  "
                    f"(t_idx={mid_t},  "
                    f"{pca.explained_variance_ratio_.sum()*100:.1f} % variance retained)",
                    fontsize=12,
                )
                plt.tight_layout()
                map_path = os.path.join(diag_dir, f"pca_map_{tag}_{vname}.png")
                plt.savefig(map_path, dpi=120, bbox_inches='tight')
                plt.close()

    def convert_xarray_to_tensor(self, ds: xr.Dataset) -> torch.Tensor:
        """Generate a tensor of data from an xarray dataset"""
        # Stacks the data variables ('pr', 'tas', ...) into a single dimension
        time_dim = getattr(self, "time_dim", "time")
        stacked_ds = ds.to_stacked_array(
            new_dim="var", sample_dims=[time_dim, "lon", "lat"]
        ).transpose("var", time_dim, "lat", "lon")
        # Convert the numpy array to a torch tensor
        tensor_data = torch.tensor(stacked_ds.to_numpy(), dtype=torch.float32)

        return tensor_data
    def get_cond_from_coords(self, coord_dict):
        years = coord_dict[self.time_dim]
        ds = self.dataset_cond.sel({self.time_dim: years})
        tensor = self.convert_xarray_to_tensor(ds)
        # Apply the already-fitted conditioning PCA if available
        if self._pca_cond is not None:
            tensor, _ = pca_denoise_dataset(
                tensor,
                n_components=self.n_components_cond,
                var_names=self.cond_vars,
                pca_objects=self._pca_cond,
            )
        return tensor

    def get_pca_state(self) -> dict:
        """Return the fitted PCA objects so they can be saved alongside a
        checkpoint and restored for consistent generation.

        Returns a dict with keys ``'target'`` and ``'cond'``, each holding
        a list of :class:`sklearn.decomposition.PCA` objects (or ``None``).
        """
        return {
            "target": self._pca_target,
            "cond": self._pca_cond,
        }

    def set_pca_state(self, state: dict) -> None:
        """Restore PCA objects from a previously saved state dict.

        Call this before :meth:`load_data` when loading a checkpoint for
        generation so that the same projection is used as during training.
        """
        self._pca_target = state.get("target")
        self._pca_cond = state.get("cond")

    def get_baseline_mean(
        self,
        baseline_start: int = 1850,
        baseline_end: int = 1900,
    ) -> torch.Tensor:
        """Compute the 1850-1900 climatological mean in *normalised* model space.

        The mean is computed from the currently loaded ``self.tensor_data``
        (shape ``[n_vars, T, H, W]``) by selecting the time indices that
        correspond to years in ``[baseline_start, baseline_end]``.

        The tensor is already in normalised space (post-preprocessing +
        normalisation, optionally PCA-filtered), so the result is directly
        comparable to model outputs and targets during training.

        Returns
        -------
        torch.Tensor
            Shape ``[1, n_vars, 1, H, W]`` — broadcast-ready for
            ``[B, n_vars, T, H, W]`` tensors used in training.

        Raises
        ------
        RuntimeError
            If no target data is loaded (``cond_only=True``) or if the loaded
            dataset contains no years in the requested baseline window.
        """
        if self.tensor_data is None or not hasattr(self, "_time_values") or self._time_values is None:
            raise RuntimeError(
                "get_baseline_mean() requires target data to be loaded.  "
                "Call load_data() first and make sure cond_only=False."
            )

        all_years = self._time_values
        mask = (all_years >= baseline_start) & (all_years <= baseline_end)

        if not mask.any():
            raise RuntimeError(
                f"No years found in the baseline window "
                f"[{baseline_start}, {baseline_end}].  "
                f"Dataset years run from {all_years.min()} to {all_years.max()}."
            )

        # tensor_data: [n_vars, T, H, W]
        indices = torch.from_numpy(np.where(mask)[0])
        baseline_tensor = self.tensor_data[:, indices, :, :]  # [n_vars, T_base, H, W]
        mean = baseline_tensor.mean(dim=1, keepdim=True)      # [n_vars, 1, H, W]
        mean = mean.unsqueeze(0)                               # [1, n_vars, 1, H, W]

        n_years = int(mask.sum())
        print(
            f"[BASELINE] Computed climatological mean over {n_years} years "
            f"({baseline_start}–{baseline_end})  "
            f"shape={tuple(mean.shape)}  "
            f"mean={mean.mean().item():.4f}  std={mean.std().item():.4f}"
        )
        return mean.float()

    def convert_tensor_to_xarray(
        self, tensor: torch.Tensor, coords: xr.DataArray = None
    ) -> xr.Dataset:
        """Generate an xarray dataset from a tensor of data"""

        assert len(tensor.shape) == 4, "Tensor must have shape (var, time, lat, lon)"

        np_data = tensor.cpu().numpy()

        # Convert the numpy array to a dictionary of xr.DataArrays
        # with the same names as the original dataset
        data_vars = {
            var_name: (["time", "lat", "lon"], np_data[i])
            for i, var_name in enumerate(self.xr_data.data_vars.keys())
        }

        # Create the dataset with the same coordinates as the original dataset
        # Note: The original time values are lost and just start at 0 instead
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": np.arange(np_data.shape[1]),
                "lat": np.linspace(-90, 90, np_data.shape[2]),
                "lon": np.linspace(0, 360, np_data.shape[3]),
            },
        ).map(denorm)

        # If we are provided time coords, create a new time coordinate
        if coords is not None:
            ds = ds.assign_coords(coords)
        return ds

    def __len__(self):
        if self.cond_only:
            return len(self.dataset_cond[self.time_dim]) - self.seq_len + 1
        return len(self.xr_data[self.time_dim]) - self.seq_len + 1

    def _append_prev_target_channel(self):
        """Append the previous timestep's first target variable to the cond tensor.

        Gives the model an explicit state in which to carry memory — the AR-1
        term that f=1 otherwise makes impossible. Measured motivation
        (scripts/diag_interannual.py): the emulator's global-mean temperature
        has lag-1 autocorrelation -0.028 against CESM2's +0.435. CESM2's lag-2
        is only +0.077, far below the 0.189 an AR(1) with phi=0.435 implies, so
        ONE previous step carries essentially all of the memory and a single
        extra channel is the right size of fix.

        THE TIME AXIS MUST BE EVENLY SPACED for this to mean "previous year".
        _select_years subsamples training to every 5th historical year and every
        other future year, so shifting by one ELEMENT there would quietly give a
        5-year lag and the model would learn the wrong memory. That is checked
        and raises rather than training on a silent mistake.

        The first timestep has no predecessor and keeps its own field
        (persistence). Zeros would be far out of distribution for a normalised
        map and would teach the model that "no history" looks like a uniform
        anomaly everywhere.

        THIS IS TEACHER FORCING: the true previous state is supplied at training
        time. At inference the model must be fed its OWN previous output, which
        is a separate change in the eval loop; the gap between the two is the
        usual exposure-bias risk, and this project has already seen the sampled
        level drift without any feedback loop at all
        (see memory: training_not_diverging_eval_noise).
        """
        if self.tensor_data is None:
            raise RuntimeError(
                "prev_target_channel needs the target tensor; it cannot be "
                "combined with cond_only=True"
            )
        t = getattr(self, "_time_values", None)
        if t is not None and len(t) > 1:
            steps = np.diff(np.asarray(t))
            if not (steps == steps[0]).all():
                raise ValueError(
                    "prev_target_channel requires an evenly spaced time axis; "
                    f"got irregular steps {sorted(set(steps.tolist()))[:5]}. "
                    "Training subsamples years (_select_years) — override it to "
                    "a contiguous range before enabling this."
                )
            if steps[0] != 1:
                print(f"[DATASET] prev_target_channel: time step is {steps[0]}, "
                      f"not 1 — the 'previous' state is {steps[0]} steps back.")

        prev = self.tensor_data[0:1].clone()          # (1, T, H, W), first target
        prev[:, 1:] = self.tensor_data[0:1, :-1]      # shift forward one step
        # prev[:, 0] keeps its own value: persistence at the boundary
        n_before = self.tensor_data_cond.shape[0]
        self.tensor_data_cond = torch.cat(
            [self.tensor_data_cond, prev], dim=0
        ).contiguous()
        print(f"[DATASET] prev_target_channel ON: cond channels {n_before} -> "
              f"{self.tensor_data_cond.shape[0]} "
              f"(appended previous-step {self.vars[0]})")

    def __getitem__(self, idx: int):
        """Defines how to get a specific index from the dataset"""
        if self.cond_only:
            raise RuntimeError(
                "ClimateDataset was created with cond_only=True — "
                "target tensor_data is not available for iteration."
            )
        return self.tensor_data[:, idx : idx + self.seq_len], self.tensor_data_cond[:, idx : idx + self.seq_len]


class EvalClimateDataset(ClimateDataset):
    """ClimateDataset that loads EVERY year, for evaluation and generation.

    Identical to :class:`ClimateDataset` in every respect except year coverage.
    Training subsamples — every 5th historical year and every other future year,
    ~76 of 251 (see :meth:`ClimateDataset._select_years`) — which is fine for
    fitting but leaves gaps when generating a continuous timeseries: a run asking
    for 1850..2014 gets 33 years spaced 5 apart.

    This class takes all years in ``[year_min, year_max]``. The chunk files on
    disk already contain them, so nothing else changes: same normalisation, same
    smoothing, same PCA, same tensor layout. Only more timesteps are loaded, so
    memory and load time scale roughly with the extra coverage (~3.3x for the
    full range).

    Do NOT use this for training — the model was fitted on the subsampled years,
    and changing coverage changes the effective sampling distribution.

    Example
    -------
        ds = EvalClimateDataset(
            seq_len=1,
            realizations=["LE2-1001.001"],
            data_dir=".../training_data/TREFHT/hist",
            target_vars=["TREFHT"],
            cond_file=".../emissions_hist_only_timefixed_bc.nc",
            cond_vars=["CO2", "SUL", "BC"],
        )
        ds.load_data("LE2-1001.001")
        ds._time_values          # every year present in the files

    Note that for CONDITIONING-only work (no CESM2 target needed, e.g. the CMIP7
    scenarios) ``eval_aero.build_cond_tensor`` reads the cond NetCDF directly and
    is simpler and cheaper than going through a dataset at all.
    """

    #: Inclusive year bounds. Widen/narrow per instance if needed.
    YEAR_MIN = 1850
    YEAR_MAX = 2100

    def __init__(self, *args, year_min: int = None, year_max: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        if year_min is not None:
            self.YEAR_MIN = int(year_min)
        if year_max is not None:
            self.YEAR_MAX = int(year_max)

    def _select_years(self) -> set:
        """Every year in [YEAR_MIN, YEAR_MAX].

        load_data() intersects this with what each file actually holds, so a
        scenario covering only part of the range is handled without error.
        """
        return set(range(self.YEAR_MIN, self.YEAR_MAX + 1))


class StratifiedPeriodSampler:
    """Batch sampler that guarantees every batch contains samples from all
    three climate periods: historical, present-day, and future.

    This forces the model to see large emission contrasts within every
    batch, providing a strong contrastive gradient signal for the
    conditioning encoder — even when training on a single scenario (e.g.
    SSP370) where the overall CO2 trend is monotonic.

    Period boundaries (default):
        historical  : year <  1950   (low CO2, near-zero anomaly)
        present     : 1950 ≤ year < 2020   (moderate CO2)
        future      : year ≥ 2020   (high CO2, large anomaly)

    Args:
        dataset:            A loaded ClimateDataset instance.
        batch_size:         Must be divisible by 3 (one third per period).
                            If not, it is rounded down to the nearest
                            multiple of 3 with a warning.
        period_boundaries:  Tuple (y1, y2) splitting the timeline into
                            historical / present / future.
        shuffle:            Whether to shuffle within each period every
                            epoch (default True).
    """

    def __init__(
        self,
        dataset: "ClimateDataset",
        batch_size: int,
        period_boundaries: tuple[int, int] = (1950, 2020),
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.shuffle = shuffle

        # Ensure batch_size is divisible by 3
        if batch_size % 3 != 0:
            batch_size = (batch_size // 3) * 3
            print(
                f"[STRATIFIED] batch_size rounded down to {batch_size} "
                f"(must be divisible by 3)"
            )
        self.batch_size = batch_size
        self.per_period = batch_size // 3

        self.y1, self.y2 = period_boundaries
        # Index arrays are built lazily when the dataset loads a realization
        self._hist_idx: Optional[np.ndarray] = None
        self._pres_idx: Optional[np.ndarray] = None
        self._fut_idx:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _build_indices(self):
        """Rebuild period index arrays from the currently loaded dataset.

        Called automatically by generate() after each load_data() call.
        Uses self.dataset.xr_data.year to assign each window index to a
        period based on the *first* year of that window.
        """
        years = self.dataset._time_values
        n_windows = len(self.dataset)   # = len(years) - seq_len + 1

        # Year of the first time-step in each window
        window_years = years[:n_windows]

        self._hist_idx = np.where(window_years <  self.y1)[0]
        self._pres_idx = np.where((window_years >= self.y1) & (window_years < self.y2))[0]
        self._fut_idx  = np.where(window_years >= self.y2)[0]

        counts = (len(self._hist_idx), len(self._pres_idx), len(self._fut_idx))
        print(
            f"[STRATIFIED] historical={counts[0]}  "
            f"present={counts[1]}  future={counts[2]}  "
            f"per_period={self.per_period}  batch_size={self.batch_size}"
        )

        if any(c == 0 for c in counts):
            missing = ["historical", "present", "future"][
                [len(self._hist_idx), len(self._pres_idx), len(self._fut_idx)].index(0)
            ]
            print(
                f"[STRATIFIED] WARNING: no samples in '{missing}' period — "
                f"falling back to uniform random sampling for this realization."
            )
            self._hist_idx = self._pres_idx = self._fut_idx = None

    # ------------------------------------------------------------------
    def _iter_batches(self) -> list[list[int]]:
        """Return a list of stratified batch index lists for one epoch."""
        if self._hist_idx is None:
            # Fallback: plain random batches
            n = len(self.dataset)
            idx = np.random.permutation(n) if self.shuffle else np.arange(n)
            return [
                idx[i : i + self.batch_size].tolist()
                for i in range(0, n - self.batch_size + 1, self.batch_size)
            ]

        def _sample_period(arr: np.ndarray, n: int) -> np.ndarray:
            """Draw n indices from arr, tiling if arr is smaller than n."""
            if self.shuffle:
                arr = np.random.permutation(arr)
            if len(arr) < n:
                arr = np.tile(arr, (n // len(arr) + 1))
            return arr[:n]

        n_batches = min(
            len(self._hist_idx), len(self._pres_idx), len(self._fut_idx)
        ) // self.per_period

        h = _sample_period(self._hist_idx, n_batches * self.per_period)
        p = _sample_period(self._pres_idx, n_batches * self.per_period)
        f = _sample_period(self._fut_idx,  n_batches * self.per_period)

        batches = []
        for i in range(n_batches):
            s, e = i * self.per_period, (i + 1) * self.per_period
            batch = np.concatenate([h[s:e], p[s:e], f[s:e]])
            if self.shuffle:
                np.random.shuffle(batch)
            batches.append(batch.tolist())
        return batches


class ClimateDataLoader:
    """DataLoader wrapper that iterates over all realizations.

    Args:
        dataset:            ClimateDataset instance.
        accelerator:        HuggingFace Accelerator.
        batch_size:         Samples per batch.
        stratified:         If True, use StratifiedPeriodSampler so every
                            batch contains historical / present / future
                            samples.  Recommended when training on a single
                            emission scenario.  Default: False.
        period_boundaries:  Passed to StratifiedPeriodSampler when
                            stratified=True.  Default: (1950, 2020).
        **dataloader_kwargs: Forwarded to torch.utils.data.DataLoader.
    """

    def __init__(
        self,
        dataset: ClimateDataset,
        accelerator: Accelerator,
        batch_size: int,
        stratified: bool = False,
        period_boundaries: tuple[int, int] = (1950, 2020),
        **dataloader_kwargs: dict[str, Any],
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.stratified = stratified
        self.dataloader_kwargs = dataloader_kwargs

        self.sampler = (
            StratifiedPeriodSampler(
                dataset,
                batch_size=batch_size,
                period_boundaries=period_boundaries,
            )
            if stratified
            else None
        )

        if stratified:
            print(
                f"[DATALOADER] Stratified period sampling enabled  "
                f"boundaries={period_boundaries}  batch_size={batch_size}"
            )

    def __len__(self):
        return self.dataset.estimate_num_batches(self.batch_size)

    def generate(self) -> torch.Tensor:
        """Iterate over all realizations, yielding stratified batches."""
        random.shuffle(self.dataset.realizations)

        for realization in self.dataset.realizations:
            # Load this realization into memory
            self.dataset.load_data(realization)

            if self.stratified and self.sampler is not None:
                # Rebuild period index arrays for the newly loaded realization
                self.sampler._build_indices()
                batches = self.sampler._iter_batches()

                for batch_indices in batches:
                    # Manually collate the indexed samples and move to device
                    samples = [self.dataset[i] for i in batch_indices]
                    batch_data = torch.stack([s[0] for s in samples]).to(self.accelerator.device)
                    batch_cond = torch.stack([s[1] for s in samples]).to(self.accelerator.device)
                    yield batch_data, batch_cond
            else:
                # Original behaviour: plain DataLoader
                dl = self.accelerator.prepare(
                    DataLoader(
                        self.dataset,
                        batch_size=self.batch_size,
                        **self.dataloader_kwargs,
                    )
                )
                for sample in dl:
                    yield sample