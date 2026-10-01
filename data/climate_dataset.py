import lumi_paths as L
import os
import random
from typing import Any, Optional, Union

from omegaconf import OmegaConf
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

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
# The _co2fix set. These four are opened by _get_processed_minmax() to derive
# the min/max anchors, so they must exist on any path that does NOT get a range
# injected from a checkpoint. A FRESH run is exactly that path, which is why
# naming the pre-co2fix files -- deleted 2026-09-07 -- killed the first asinh
# launch at the first batch with FileNotFoundError while every RESUMED run kept
# working, the persisted COND_NORM short-circuiting the lookup before it opened
# anything.
EMISSIONS_PATHS = [
    f"{L.DATA}/emissions_hist_only_timefixed_bc_co2fix.nc",
    f"{L.DATA}/emissions_ssp370_only_timefixed_bc_co2fix.nc",
    f"{L.DATA}/emissions_aaer_only_timefixed_bc_co2fix.nc",
    f"{L.DATA}/emissions_ghg_only_timefixed_bc_co2fix.nc",
]

# Which of those files the ANCHORS are fitted on (the files the DATASET reads
# are set per experiment in the data config and are unaffected).
#
#   "all4"        every shipped checkpoint: hist + ssp370 + aaer + ghg pooled.
#   "hist_ssp370" the two scenarios that actually exercise every channel.
#
# WHY IT MATTERS: ghg holds SUL/BC at ~0 and aaer holds CO2 near pre-industrial,
# so pooling all four pulls each channel's percentile toward a scenario that
# does not drive it. Measured: CO2's hi anchor is 4.30e-02 over all four vs
# 9.94e-02 over hist+ssp370 -- a factor 2.3, and since the anchor DIVIDES, the
# model's input is 2.3x larger under "all4" for the same emissions.
#
# Changing this changes the meaning of every cond channel => FRESH run only.
# Persisted per checkpoint (COND_NORM for the raw path, COND_PROCESSED_NORM for
# normalize_last) so eval reproduces training exactly instead of refitting.
_ANCHOR_SCENARIOS = "all4"


def set_anchor_scenarios(which: str) -> None:
    global _ANCHOR_SCENARIOS, _PROCESSED_MINMAX_CACHE
    which = str(which).lower()
    if which not in ("all4", "hist_ssp370"):
        raise ValueError(f"anchor_scenarios must be all4|hist_ssp370, got {which!r}")
    _ANCHOR_SCENARIOS = which
    _PROCESSED_MINMAX_CACHE = None
    print(f"[COND] anchor scenarios = {which}")


def get_anchor_scenarios() -> str:
    return _ANCHOR_SCENARIOS


def _anchor_paths():
    """The subset of EMISSIONS_PATHS the anchors are fitted on."""
    if _ANCHOR_SCENARIOS == "hist_ssp370":
        return [p for p in EMISSIONS_PATHS
                if ("_hist_" in p or "_ssp370_" in p)]
    return list(EMISSIONS_PATHS)


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


# ---------------------------------------------------------------------------


# =============================================================================
# Active cond-normalisation path (used by the training pipeline)
# =============================================================================

# The conditioning pipeline, in full:
#
#     cond_file -> gaussian smooth (cond_smooth_sigma) -> min/max normalise
#
# ONE transform, no clip, no PCA. The anchors are the true min and max of the
# SMOOTHED reference field, so +1 is the largest cell that exists in training
# and nothing can leave [-1, +1] without a clip. Fitted on hist+ssp370 only
# (anchor_scenarios): ghg holds SUL/BC at ~0 and aaer holds CO2 near
# pre-industrial, so pooling all four drags each channel's anchor toward a
# scenario that does not drive it.
#
# WHY THIS AND NOT THE ALTERNATIVES (all measured, all removed 2026-10-01):
#   percentile + clip ("v1")  pinned whole countries at +1 -- E China BC was
#       100% of years saturated -- destroying their industrial history before
#       the model saw it, and produced grid-scale speckle in the output.
#   asinh                     compressed the tail but amplified near-zero
#       cells, leaking attribution onto oceans.
#   minmax + asinh stretch    fixed the bulk-contrast cost but retained only
#       20% of the late Asian rise against truth's 71%; the linear map keeps
#       70%, which is why this one won.
#
# The anchors in force are persisted per checkpoint as COND_PROCESSED_NORM and
# re-injected at eval, so an eval reproduces training exactly instead of
# refitting and trusting the two to agree.
_PROCESSED_MINMAX_CACHE: "dict | None" = None
_PROCESSED_OVERRIDE: "dict | None" = None


def set_processed_minmax_override(anchors: "dict | None") -> None:
    """Inject checkpoint-persisted anchors (see COND_PROCESSED_NORM).

    Pass None to CLEAR: a process loading a second checkpoint must not inherit
    the first one's anchors. Refitting instead of injecting is what let the
    first normalize_last evals normalise SUL ~6x off in silence.
    """
    global _PROCESSED_OVERRIDE, _PROCESSED_MINMAX_CACHE
    _PROCESSED_OVERRIDE = (None if anchors is None else
                           {str(k): (float(v[0]), float(v[1])) for k, v in anchors.items()})
    _PROCESSED_MINMAX_CACHE = None


def get_processed_minmax_state() -> "dict | None":
    """The anchors actually in force, for persisting to a checkpoint."""
    return _PROCESSED_MINMAX_CACHE


def _get_processed_minmax(sigmas, var_names):
    """Min/max anchors fitted on the SMOOTHED reference fields.

    Mirrors the real pipeline: per reference scenario, smooth each channel with
    its own sigma, then pool across scenarios and take the true min and max.

    The anchors MUST come from the smoothed field. Smoothing is linear and so
    commutes with this affine map, but only if hi is taken after smoothing;
    taken from the raw field a single unsmoothed hotspot sets the ceiling and
    the smoothed field never approaches +1 (SUL's raw max is ~60x its smoothed
    one).
    """
    global _PROCESSED_MINMAX_CACHE
    if _PROCESSED_MINMAX_CACHE is not None:
        return _PROCESSED_MINMAX_CACHE
    if _PROCESSED_OVERRIDE is not None:
        print(f"[COND] processed anchors from checkpoint: {_PROCESSED_OVERRIDE}")
        _PROCESSED_MINMAX_CACHE = _PROCESSED_OVERRIDE
        return _PROCESSED_MINMAX_CACHE
    pooled = {v: [] for v in var_names}
    for path in _anchor_paths():
        ds_emis = xr.open_dataset(path)
        for v_idx, var in enumerate(var_names):
            if var not in ds_emis.data_vars:
                continue
            arr = np.asarray(ds_emis[var].values, dtype=np.float64)   # (T, H, W)
            sig = float(sigmas[v_idx]) if sigmas is not None else 0.0
            if sig > 0:
                arr = smooth_cond_spatial(arr[None], [sig], "gaussian", [var])[0]
            pooled[var].append(arr.ravel())
        ds_emis.close()
    out = {}
    for var, arrays in pooled.items():
        if not arrays:
            continue
        flat = np.concatenate(arrays)
        flat = flat[np.isfinite(flat)]
        sig_v = sigmas[var_names.index(var)] if sigmas else 0
        out[var] = (float(flat.min()), float(flat.max()))
        print(f"[COND] processed anchors for {var} (sigma={sig_v}, MIN-MAX): "
              f"lo={out[var][0]:.4e} hi={out[var][1]:.4e}")
    _PROCESSED_MINMAX_CACHE = out
    return out


def normalize_tensor_cond(tensor, var_names, sigmas):
    """Min/max normalise an ALREADY smoothed cond tensor to [-1, +1]."""
    anchors = _get_processed_minmax(sigmas, var_names)
    out = tensor.clone()
    for v_idx, var in enumerate(var_names):
        lo_, hi_ = anchors[var]
        if hi_ <= lo_:
            out[v_idx] = -1.0
            continue
        mid = (lo_ + hi_) / 2.0
        half = (hi_ - lo_) / 2.0
        out[v_idx] = (tensor[v_idx] - mid) / half
        print(f"[COND] cond {var}: lo={lo_:.4e} hi={hi_:.4e} "
              f"-> range [{float(out[v_idx].min()):.2f}, {float(out[v_idx].max()):.2f}]")
    return torch.nan_to_num(out, nan=-1.0)


def normalize(ds: xr.DataArray) -> xr.DataArray:
    """Normalise a TARGET DataArray (TREFHT, PRECT) via the fixed NORM_FN maps.

    Conditioning channels do NOT pass through here: they are smoothed first and
    normalised as a tensor by normalize_tensor_cond().
    """
    if ds.name in ("CO2", "SUL", "BC"):
        raise RuntimeError(
            f"normalize() was called on cond channel {ds.name!r}. Cond is "
            f"normalised after smoothing by normalize_tensor_cond(); routing it "
            f"through here would normalise the RAW inventory instead."
        )
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
        # Per-channel spatial gaussian σ (in gridpoints) applied to normalised
        # cond fields.  None or 0 disables smoothing for that channel.
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
        # RAW values: smoothing runs first, normalize_tensor_cond applies the
        # min/max map afterwards with the anchors fitted on the smoothed field.
        raw_cond = raw_cond[self.cond_vars]

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

        # ── Spatial smoothing on conditioning (before normalisation) ─────────
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

        # ── Normalisation, AFTER smoothing ───────────────────────────────────
        self.tensor_data_cond = normalize_tensor_cond(
            self.tensor_data_cond, self.cond_vars,
            self.cond_smooth_sigma).contiguous()

        # Save diagnostic spatial plots (only on first load)
        diag_dir = os.path.join(self.data_dir, "diagnostics")
        if not os.path.isdir(diag_dir):
            os.makedirs(diag_dir, exist_ok=True)
            self._save_cond_diagnostics(diag_dir)

    def _save_cond_diagnostics(self, diag_dir: str):
        """Save spatial maps and time series of conditioning data.

        The plots show the tensor the model actually receives: smoothed, then
        min/max normalised.
        """
        all_years = self.dataset_cond[self.time_dim].values
        candidate_years = [all_years[0], 2015, 2050, all_years[-1]]
        years_to_show   = [y for y in candidate_years if y in all_years]
        year_indices    = [int(np.where(all_years == y)[0][0]) for y in years_to_show]

        # tensor_data_cond shape: (n_vars, T, H, W) — exactly what the model sees
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

            fig.suptitle(
                f"{var} — cond_map seen by model",
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

            ax.set_title(
                f"{var} — spatial mean of cond_map seen by model\n"
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
        return tensor

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
        normalisation), so the result is directly
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
    smoothing, same tensor layout. Only more timesteps are loaded, so
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