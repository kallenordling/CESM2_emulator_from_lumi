#!/usr/bin/env python3
"""eval_aero.py
==============
Evaluate the multi-experiment CESM2 aerosol emulator.

Loads the latest checkpoint from runs/, generates temperature time series
for all 4 experiments (hist, ssp370, aaer, ghg), and produces:
  1. global_mean_anomaly.png  — area-weighted global mean anomaly (re 1850–1900)
     for all experiments, model vs CESM2 ensemble member
  2. anomaly_maps_<scenario>.png — spatial anomaly maps at key years per experiment

Run from the project root:
    python eval_aero.py [--runs-dir /path/to/runs] [--output-dir eval_output]
"""

import argparse
import contextlib
import os
import re
import sys
import glob

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy import stats
from omegaconf import OmegaConf
from hydra.utils import instantiate
from ema_pytorch import EMA
from tqdm import tqdm

# ── project imports ────────────────────────────────────────────────────────────
from data.climate_dataset import (
    ClimateDataset,
    normalize,
    DENORM_FN,
    PREPROCESS_FN,
    pca_denoise_dataset,
)
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from models.video_net import UNetModel3D

# ── paths ──────────────────────────────────────────────────────────────────────
PROJ_ROOT   = "/projappl/project_462001328/CESM2_emulator_from_lumi"
SCRATCH     = "/scratch/project_462001328/emulator_data"
RUNS_DIR    = os.path.join(PROJ_ROOT, "runs")
DATA_ROOT   = os.path.join(SCRATCH, "training_data/TREFHT")
EMIS_DIR    = SCRATCH
CONFIG_PATH = "configs/config_aero.yaml"

# ── experiment definitions ─────────────────────────────────────────────────────
EXPERIMENTS = [
    dict(
        name         = "hist",
        data_dir     = os.path.join(DATA_ROOT, "hist"),
        cond_file    = os.path.join(EMIS_DIR, "emissions_hist_timefixed.nc"),
        realizations = [
            "LE2-1001.001", "LE2-1011.001", "LE2-1021.002", "LE2-1031.002",
            "LE2-1041.003", "LE2-1051.003", "LE2-1061.004", "LE2-1071.004",
            "LE2-1081.005", "LE2-1091.005",
        ],
        time_dim     = "time",
        map_years    = [1900, 2000, 2014],   # last available year instead of 2100
        color        = "#1f77b4",
    ),
    dict(
        name         = "ssp370",
        data_dir     = os.path.join(DATA_ROOT, "ssp370"),
        cond_file    = os.path.join(EMIS_DIR, "emissions_ssp370_timefixed.nc"),
        realizations = [
            "LE2-1001.001", "LE2-1011.001", "LE2-1021.002", "LE2-1031.002",
            "LE2-1041.003", "LE2-1051.003", "LE2-1061.004", "LE2-1071.004",
            "LE2-1081.005", "LE2-1091.005",
        ],
        time_dim     = "time",
        map_years    = [2015, 2050, 2100],
        color        = "#d62728",
    ),
    dict(
        name         = "ssp126",
        # Pre-regridded CESM2 ssp126 monthly tas (K), single member r4i1p1f1.
        data_dir     = os.path.join(SCRATCH, "cmip6", "CESM2_ssp126.nc"),
        # ssp126-only cond file (2015–2100); cumulative CO2 still integrated
        # from 1850 so magnitudes match the training distribution.
        cond_file    = os.path.join(EMIS_DIR, "emissions_ssp126_only_timefixed.nc"),
        realizations = ["r4i1p1f1"],
        time_dim     = "time",
        target_var   = "tas",
        map_years    = [2015, 2050, 2100],
        color        = "#9467bd",
    ),
    dict(
        name         = "aaer",
        data_dir     = os.path.join(DATA_ROOT, "AAER"),
        cond_file    = os.path.join(EMIS_DIR, "emissions_aero_only_timefixed.nc"),
        realizations = ["001", "002", "003", "004", "005",
                        "006", "007", "008", "009", "010"],
        time_dim     = "time",
        map_years    = [1900, 2000, 2050],
        color        = "#ff7f0e",
    ),
    dict(
        name         = "ghg",
        data_dir     = os.path.join(DATA_ROOT, "GHG"),
        cond_file    = os.path.join(EMIS_DIR, "emissions_ghg_only_timefixed.nc"),
        realizations = ["001", "002", "003", "004", "005",
                        "006", "007", "008", "009", "010"],
        time_dim     = "time",
        map_years    = [1900, 2000, 2050],
        color        = "#2ca02c",
    ),
]

BASELINE_START = 1850
BASELINE_END   = 1900

# Two training members used as an internal-variability reference.
# Their ΔT difference is shown as a grey band on bias panels so the
# reader can judge whether model bias is within natural variability.
# Must NOT overlap with the held-out validation member (LE2-1231.001).
REF_REALIZATIONS = {
    "hist":   ["LE2-1011.001", "LE2-1021.002"],
    "ssp370": ["LE2-1011.001", "LE2-1021.002"],
}
SAMPLE_STEPS   = 50           # fewer steps than training → faster inference
BATCH_SIZE     = 16           # years per GPU batch
N_ENSEMBLE     = 1            # diffusion samples per experiment
COND_VARS      = ["CO2", "SUL"]
TARGET_VAR     = "TREFHT"
LAT  = None   # set from first conditioning file in main()
LON  = None   # set from first conditioning file in main()

NULL_COND = -1.0   # CFG null value (pre-industrial baseline under normalisation)

# Per-channel anti-guidance scales (< 1.0 reduces overcounting, 1.0 = no change).
# Set both to 1.0 to disable and use direct conditioning (original behaviour).
GUIDANCE_CO2 = 1.0
GUIDANCE_SUL = 1.0

# Time windows for spatial IG maps, keyed by experiment name
IG_WINDOWS = {
    "hist":   [(1920, 1960, "1920–1960"), (1960, 1990, "1960–1990"), (1990, 2014, "1990–2014")],
    "ssp370": [(2020, 2050, "2020–2050"), (2050, 2080, "2050–2080"), (2080, 2100, "2080–2100")],
    "ssp126": [(2020, 2050, "2020–2050"), (2050, 2080, "2050–2080"), (2080, 2100, "2080–2100")],
    "aaer":   [(1920, 1960, "1920–1960"), (1960, 1990, "1960–1990"), (1990, 2100, "1990–2100")],
    "ghg":    [(1920, 1970, "1920–1970"), (1970, 2020, "1970–2020"), (2050, 2100, "2050–2100")],
}

# Representative output locations for Option-C IG (name, lat°N, lon°E)
OUTPUT_LOCATIONS = [
    ("Arctic",     85,   0),
    ("N.America",  45, 260),
    ("Europe",     50,  10),
    ("E.Asia",     35, 120),
    ("Tropics",     0, 170),
    ("Antarctic", -70,   0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(runs_dir: str) -> str:
    """Return path to the highest-epoch checkpoint in runs_dir."""
    paths = [p for p in glob.glob(os.path.join(runs_dir, "*.pt"))
             if not p.endswith("_best.pt")]
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {runs_dir}")

    def _epoch(p):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    best = max(paths, key=_epoch)
    print(f"[CHECKPOINT] Using: {best}  (epoch {_epoch(best)})")
    return best


def load_model(ckpt_path: str, config_path: str, device: torch.device):
    """Load UNet model from checkpoint, return (model, pca_state)."""
    cfg = OmegaConf.load(config_path)
    model: UNetModel3D = instantiate(cfg.model)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[MODEL] Checkpoint keys: {list(ckpt.keys())}")
    print(f"[MODEL] Global step: {ckpt.get('Global Step', 'n/a')}")

    missing, unexpected = model.load_state_dict(ckpt["EMA"], strict=False)
    if missing:
        print(f"[MODEL] {len(missing)} missing keys — zero-initialised")
        for k in missing[:5]:
            print(f"  {k}")
    if unexpected:
        print(f"[MODEL] {len(unexpected)} unexpected keys (old arch)")

    model = model.to(device).eval()
    pca_state = ckpt.get("PCA")
    return model, pca_state


def extract_years(coord_vals) -> np.ndarray:
    """Extract integer years from cftime or integer coordinate array."""
    if hasattr(coord_vals[0], "year"):
        return np.array([int(str(v)[:4]) for v in coord_vals])
    return np.asarray(coord_vals, dtype=int)


def build_cond_tensor(cond_file: str, cond_vars: list, time_dim: str,
                      pca_objects, n_components_cond, cond_smooth_sigma=None):
    """Load, normalize, optionally smooth + PCA-project the conditioning data.

    The smoothing + PCA steps must mirror the training-side pipeline in
    ClimateDataset exactly — otherwise the model is fed raw, spiky inventory
    fields at inference that it never saw in training, and it imprints the
    grid-scale texture (shipping lanes, flight paths) onto the output.

    Returns:
        cond_tensor : torch.Tensor  (n_vars, T, H, W)
        years       : np.ndarray    (T,) integer years
        lat         : np.ndarray    (H,) latitude values from file
        lon         : np.ndarray    (W,) longitude values from file
    """
    raw = xr.open_dataset(cond_file)
    # Some cond files (e.g. emissions_ghg_only_timefixed.nc) store the time axis
    # as 'year'; the training-side loader renames it on open (78d5364).
    # Mirror that here so downstream stacking/chunking can rely on `time_dim`.
    if time_dim not in raw.dims and "year" in raw.dims:
        raw = raw.rename({"year": time_dim})
    raw = raw[cond_vars].chunk({time_dim: -1})
    norm = raw.map(normalize)

    lat = norm["lat"].values.astype(np.float64)
    lon = norm["lon"].values.astype(np.float64)

    # to_stacked_array needs ("var", time_dim, "lat", "lon")
    stacked = norm.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"])
    stacked = stacked.transpose("var", time_dim, "lat", "lon")
    cond_tensor = torch.tensor(stacked.values, dtype=torch.float32)

    years = extract_years(norm[time_dim].values)

    # ── Spatial smoothing on conditioning (before PCA) ───────────────────────
    # Mirrors ClimateDataset (data/climate_dataset.py): per-channel Gaussian
    # smoothing removes inventory line features (SO2 shipping lanes, flight
    # paths) so the model receives the same smooth cond it was trained on.
    # Longitude axis uses wrap padding (periodic); latitude uses reflect.
    if cond_smooth_sigma is not None:
        from scipy.ndimage import gaussian_filter1d
        if isinstance(cond_smooth_sigma, (int, float)):
            sigmas = [float(cond_smooth_sigma)] * len(cond_vars)
        else:
            sigmas = [float(s) for s in cond_smooth_sigma]
        arr = cond_tensor.numpy()                   # (n_vars, T, H, W)
        for v_idx, sigma in enumerate(sigmas):
            if sigma <= 0:
                continue
            channel = arr[v_idx]                    # (T, H, W)
            channel = gaussian_filter1d(channel, sigma=sigma, axis=-1, mode="wrap")
            channel = gaussian_filter1d(channel, sigma=sigma, axis=-2, mode="reflect")
            arr[v_idx] = channel
            print(f"[COND] spatial gaussian smoothing σ={sigma} applied to "
                  f"{cond_vars[v_idx]}")
        cond_tensor = torch.from_numpy(arr).contiguous()

    if pca_objects is not None:
        cond_tensor, _ = pca_denoise_dataset(
            cond_tensor,
            n_components=n_components_cond,
            pca_objects=pca_objects,
        )

    raw.close()
    return cond_tensor, years, lat, lon


def generate_timeseries(
    model: UNetModel3D,
    scheduler: ContinuousDDPM,
    cond_tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    sample_steps: int,
    batch_size: int = 16,
    seed: int | None = None,
    guidance_co2: float = 1.0,
    guidance_sul: float = 1.0,
    force_cfg: bool = False,
    autocast_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Diffusion sampling for every year in cond_tensor.

    Args:
        cond_tensor: (n_vars, T, H, W) normalised conditioning on CPU
        seed: optional RNG seed for reproducible ensemble members
        guidance_co2: per-channel CFG scale for CO2 (< 1.0 = anti-guidance)
        guidance_sul: per-channel CFG scale for SUL (< 1.0 = anti-guidance)
            formula: pred = pred_null + w_co2*(pred_co2 - pred_null)
                                      + w_sul*(pred_sul - pred_null)
            Both 1.0 → direct conditioning (original behaviour, 1 forward pass).
    Returns:
        numpy array (T, H, W) in *normalised* model output space
    """
    use_cfg = force_cfg or (guidance_co2 != 1.0) or (guidance_sul != 1.0)

    if seed is not None:
        torch.manual_seed(seed)
    _, T, H, W = cond_tensor.shape
    scheduler.set_timesteps(sample_steps)
    steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

    results = []
    for i in tqdm(range(0, T, batch_size), desc="  generating batches"):
        chunk = cond_tensor[:, i: i + batch_size]          # (C, B, H, W)
        B = chunk.shape[1]
        # model expects (B, C, 1, H, W)
        cond_b = chunk.permute(1, 0, 2, 3).unsqueeze(2).to(device=device, dtype=dtype)
        gen    = torch.randn(B, 1, 1, H, W, device=device, dtype=dtype)

        if use_cfg:
            cond_co2_only = cond_b.clone()
            cond_co2_only[:, 1] = NULL_COND          # SUL nulled
            cond_sul_only = cond_b.clone()
            cond_sul_only[:, 0] = NULL_COND          # CO2 nulled
            cond_null = torch.full_like(cond_b, NULL_COND)

        # Mixed-precision context: ops auto-cast to autocast_dtype where the
        # backend supports it; unsupported ones (e.g. Conv3d shapes MIOpen
        # lacks a bf16 kernel for) fall back to fp32 cleanly inside the
        # autocast region — no hard dtype mismatch.
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            for step_idx, t_idx in enumerate(scheduler.timesteps):
                t = scheduler.log_snr(steps[t_idx]).expand(B).to(dtype)
                if use_cfg:
                    pred_co2  = model(gen, t, cond_map=cond_co2_only)
                    pred_sul  = model(gen, t, cond_map=cond_sul_only)
                    pred_null = model(gen, t, cond_map=cond_null)
                    pred = (pred_null
                            + guidance_co2 * (pred_co2 - pred_null)
                            + guidance_sul * (pred_sul - pred_null))
                else:
                    pred = model(gen, t, cond_map=cond_b)
                # Scheduler step expects fp32; autocast may return bf16, so
                # cast pred back to gen's dtype before the update.
                gen  = scheduler.step(pred.to(gen.dtype), timestep=t_idx, sample=gen).prev_sample

        results.append(gen.squeeze(1).squeeze(1).cpu().float())   # (B, H, W)

    return torch.cat(results, dim=0).numpy()   # (T, H, W)


def load_cesm2_annual_single(data_dir: str, realization: str, time_dim: str,
                              target_var: str = TARGET_VAR) -> tuple:
    """Load CESM2 `target_var` for one realization, return (years, data_celsius array).

    data_celsius shape: (T, lat, lon)

    If `data_dir` is a direct .nc file path, it is opened as-is and `realization`
    is ignored (useful for pre-regridded single-member files).
    """
    if os.path.isfile(data_dir):
        path = data_dir
    else:
        path = os.path.join(data_dir, realization, "*.nc")
    ds = xr.open_mfdataset(path, combine="by_coords",
                           chunks={time_dim: 50})[target_var]

    # Convert K → °C
    ds = ds - 273.15

    # Resample to annual mean if sub-annual (monthly/daily)
    if time_dim == "time":
        try:
            ds_annual = ds.resample(time="YE").mean().compute()
            years = extract_years(ds_annual.time.values)
            data  = ds_annual.values.astype(np.float32)         # (T, lat, lon)
        except Exception:
            # fallback: assume already annual
            ds.load()
            years = extract_years(ds[time_dim].values)
            data  = ds.values.astype(np.float32)
    else:
        ds.load()
        years = extract_years(ds[time_dim].values)
        data  = ds.values.astype(np.float32)

    return years, data


def load_cesm2_ensemble(data_dir: str, realizations: list, time_dim: str,
                         target_var: str = TARGET_VAR) -> tuple:
    """Load CESM2 `target_var` for multiple realizations.

    Returns:
        years           : np.ndarray (T,) — years from first successfully loaded member
        cesm_ensemble   : np.ndarray (N, T, lat, lon) — all members on common years
    """
    members = []
    common_years = None
    for real in realizations:
        try:
            yrs, data = load_cesm2_annual_single(data_dir, real, time_dim, target_var)
            if common_years is None:
                common_years = yrs
                members.append(data)
            else:
                # align to common years
                common, idx_c, idx_m = np.intersect1d(common_years, yrs, return_indices=True)
                members = [m[idx_c] for m in members]
                members.append(data[idx_m])
                common_years = common
        except Exception as e:
            print(f"    WARNING: could not load realization {real}: {e}")

    if not members:
        raise FileNotFoundError(f"No CESM2 members loaded from {data_dir}")

    return common_years, np.stack(members, axis=0)   # (N, T, lat, lon)


def area_weighted_gmean(data: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Area-weighted global mean.  data: (..., H, W), lat: (H,)."""
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]           # (H, 1)
    w /= w.mean()
    return (data * w).mean(axis=(-2, -1))                # (...,)


def load_co2_global_annual(cond_file: str, time_dim: str, lat: np.ndarray) -> tuple:
    """Load raw (un-normalised) CO2 field and return the global annual total.

    The cond files store CO2 with units "Gt CO2 / year / gridpoint" — each
    gridcell already holds its own contribution, so the correct global total
    is a plain sum over (lat, lon), not an area-weighted mean. Cumulative
    sum over time then yields cumulative GtCO2, which is the physical axis
    used in TCRE diagrams.

    Returns
    -------
    years      : np.ndarray (T,)  integer years
    co2_annual : np.ndarray (T,)  global annual CO2 emissions (Gt CO2 / year)
    """
    ds = xr.open_dataset(cond_file)
    if time_dim not in ds.dims and "year" in ds.dims:
        ds = ds.rename({"year": time_dim})
    if "CO2" not in ds:
        ds.close()
        return None, None
    co2 = ds["CO2"].values.astype(np.float64)   # (T, H, W), GtCO2/yr/gridpoint
    years = extract_years(ds[time_dim].values)
    ds.close()
    co2_annual = co2.sum(axis=(-2, -1))          # (T,) GtCO2/yr globally
    return years, co2_annual


def plot_tcre(results: dict, out_path: str):
    """Plot ΔT vs cumulative CO2 (TCRE diagram) for hist + projections.

    X-axis : cumulative CO2 (area-weighted global mean of raw CO2 field,
             cumsummed from the start of the hist record). Each projection
             scenario gets its own hist+projection cumulative trajectory,
             so ssp370 and ssp126 diverge from the shared hist segment.
    Y-axis : global-mean temperature anomaly re 1850–1900.

    Solid lines  = model ensemble mean  (shaded spread when N > 1).
    Dashed lines = CESM2 ensemble mean  (shaded spread when N > 1).
    Linear regression slopes annotated for model and CESM2 per projection.
    """
    PROJECTIONS = [p for p in ("ssp370", "ssp126")
                   if p in results and results[p].get("co2_annual") is not None]

    if "hist" not in results or not PROJECTIONS:
        print("[TCRE] Need hist + at least one projection with co2_annual — skipping.")
        return

    # ── build per-projection cumulative CO2 lookup (hist + projection) ───
    # Each projection gets its own lookup so ssp370 and ssp126 share the
    # hist segment (1850-2014) and diverge from 2015 onwards.
    lookups = {}   # proj_name -> {year_int: cum_co2}
    for proj in PROJECTIONS:
        parts_y, parts_c = [], []
        for sc in ("hist", proj):
            d = results.get(sc, {})
            if d.get("co2_annual") is None:
                continue
            parts_y.append(d["co2_years"])
            parts_c.append(d["co2_annual"])
        if not parts_y:
            continue
        all_y = np.concatenate(parts_y)
        all_c = np.concatenate(parts_c)
        # dedupe any overlapping years, keeping the first occurrence (hist wins
        # on the 1850-2014 range by order of concatenation)
        _, keep_idx = np.unique(all_y, return_index=True)
        keep_idx = np.sort(keep_idx)
        all_y, all_c = all_y[keep_idx], all_c[keep_idx]
        order = np.argsort(all_y)
        all_y, all_c = all_y[order], all_c[order]
        lookups[proj] = dict(zip(all_y.astype(int), np.cumsum(all_c)))

    if not lookups:
        print("[TCRE] Could not build any cumulative CO2 lookup — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    ax_main, ax_bias = axes

    def get_cumco2(years, proj):
        return np.array([lookups[proj].get(int(y), np.nan) for y in years])

    # Any projection's lookup gives the same cumulative on hist's 1850-2014
    # range (shared segment), so pick the first for hist plotting.
    hist_proj = PROJECTIONS[0]

    # ── per-scenario plot ─────────────────────────────────────────────────
    # Draw hist once (shared segment), then each projection against its own
    # lookup so the X-axis accurately reflects that scenario's cumulative CO2.
    scenario_plot_list = [("hist", hist_proj)] + [(p, p) for p in PROJECTIONS]

    for sc, lookup_key in scenario_plot_list:
        d = results.get(sc)
        if d is None:
            continue
        c = d["color"]
        gen_ens  = d["gen_anom_ens"]          # (N_gen, T)
        gen_years = d["gen_years"]
        gen_mean = gen_ens.mean(axis=0)        # (T,)
        N_gen    = gen_ens.shape[0]

        cumco2 = get_cumco2(gen_years, lookup_key)
        valid  = ~np.isnan(cumco2)

        if N_gen == 1:
            ax_main.plot(cumco2[valid], gen_mean[valid], color=c, lw=1.8,
                         label=f"{sc} model")
        else:
            lo = gen_ens[:, valid].min(axis=0)
            hi = gen_ens[:, valid].max(axis=0)
            ax_main.fill_between(cumco2[valid], lo, hi, color=c, alpha=0.18)
            ax_main.plot(cumco2[valid], gen_mean[valid], color=c, lw=1.8,
                         label=f"{sc} model (N={N_gen})")

        if d.get("cesm_anom") is not None:
            cesm_ens   = d["cesm_anom_ens"]      # (N_cesm, T)
            cesm_mean  = d["cesm_anom"]           # (T,)
            cesm_years = d["cesm_years"]
            N_cesm     = cesm_ens.shape[0]

            cumco2_c = get_cumco2(cesm_years, lookup_key)
            valid_c  = ~np.isnan(cumco2_c)

            if N_cesm == 1:
                ax_main.plot(cumco2_c[valid_c], cesm_mean[valid_c],
                             color=c, lw=1.8, ls="--", alpha=0.8,
                             label=f"{sc} CESM2")
            else:
                lo_c = cesm_ens[:, valid_c].min(axis=0)
                hi_c = cesm_ens[:, valid_c].max(axis=0)
                ax_main.fill_between(cumco2_c[valid_c], lo_c, hi_c,
                                     facecolor=c, alpha=0.10, lw=0)
                ax_main.fill_between(cumco2_c[valid_c], lo_c, hi_c,
                                     facecolor="none", hatch="///",
                                     edgecolor=c, alpha=0.4, lw=0)
                ax_main.plot(cumco2_c[valid_c], cesm_mean[valid_c],
                             color=c, lw=1.8, ls="--", alpha=0.8,
                             label=f"{sc} CESM2 (N={N_cesm})")

            # bias panel: model − CESM2 on common years
            common_y, ig, ic = np.intersect1d(gen_years, cesm_years,
                                               return_indices=True)
            cumco2_common = get_cumco2(common_y, lookup_key)
            valid_b = ~np.isnan(cumco2_common)
            diff = gen_mean[ig] - cesm_mean[ic]
            ax_bias.plot(cumco2_common[valid_b], diff[valid_b],
                         color=c, lw=1.5, label=f"{sc}")
            if N_gen > 1:
                dlo = gen_ens[:, ig][:, valid_b].min(axis=0) - cesm_mean[ic][valid_b]
                dhi = gen_ens[:, ig][:, valid_b].max(axis=0) - cesm_mean[ic][valid_b]
                ax_bias.fill_between(cumco2_common[valid_b], dlo, dhi,
                                     color=c, alpha=0.15)

    # ── regression summary per projection (hist + projection combined) ────
    def _combined_regression(proj, key_ens, key_years):
        xs, ys = [], []
        for sc in ("hist", proj):
            d = results.get(sc)
            if d is None or d.get(key_ens) is None:
                continue
            yr = d[key_years]
            en = d[key_ens].mean(axis=0)
            cc = get_cumco2(yr, proj)
            v  = ~np.isnan(cc)
            xs.append(cc[v]);  ys.append(en[v])
        if not xs:
            return None, None
        xs = np.concatenate(xs);  ys = np.concatenate(ys)
        return np.polyfit(xs, ys, 1)

    fit_line_styles = {"ssp370": "-", "ssp126": "-."}
    for proj in PROJECTIONS:
        m_slope, m_int = _combined_regression(proj, "gen_anom_ens", "gen_years")
        c_slope, c_int = _combined_regression(proj, "cesm_anom_ens", "cesm_years")
        # x range: span of this projection's cumulative-CO2 trajectory
        xs_p = np.fromiter(lookups[proj].values(), dtype=float)
        x_range = np.array([xs_p.min(), xs_p.max()])
        ls = fit_line_styles.get(proj, ":")
        if m_slope is not None:
            ax_main.plot(x_range, m_slope * x_range + m_int,
                         color="k", lw=1.2, ls=ls,
                         label=f"Model fit {proj}  slope={m_slope:.4f}")
        if c_slope is not None:
            ax_main.plot(x_range, c_slope * x_range + c_int,
                         color="0.4", lw=1.2, ls=ls,
                         label=f"CESM2 fit {proj}  slope={c_slope:.4f}")
        # print the ratio so the user can read systemic TCRE bias scenario-by-scenario
        if m_slope is not None and c_slope is not None and c_slope != 0:
            print(f"  [TCRE] {proj}: model/CESM2 slope ratio = "
                  f"{m_slope / c_slope:.3f}  "
                  f"(model {m_slope:.4f}, CESM2 {c_slope:.4f})")

    ax_main.axhline(0, color="k", lw=0.6, ls=":")
    ax_main.set_xlabel("Cumulative CO₂ emissions (GtCO₂)")
    ax_main.set_ylabel("Global-mean TREFHT anomaly re 1850–1900 (°C)")
    proj_title = " + ".join(PROJECTIONS)
    ax_main.set_title(f"TCRE — ΔT vs cumulative CO₂  (hist + {proj_title})")
    ax_main.legend(fontsize=7, ncol=2)
    ax_main.grid(True, alpha=0.25)

    # grey ±|member diff| band on TCRE bias panel (internal variability reference)
    ref_band_drawn_tcre = False
    for sc, lookup_key in scenario_plot_list:
        d = results.get(sc)
        if d is None or d.get("ref_diff") is None or d.get("ref_years") is None:
            continue
        ry  = d["ref_years"]
        rd  = np.abs(d["ref_diff"])
        cc  = get_cumco2(ry, lookup_key)
        v   = ~np.isnan(cc)
        if v.any():
            label = "±|member diff| (nat. var.)" if not ref_band_drawn_tcre else None
            ax_bias.fill_between(cc[v], -rd[v], rd[v], color="grey", alpha=0.20,
                                 zorder=0, label=label)
            ref_band_drawn_tcre = True

    ax_bias.axhline(0, color="k", lw=0.9)
    ax_bias.set_xlabel("Cumulative CO₂ emissions (GtCO₂)")
    ax_bias.set_ylabel("Bias: model − CESM2 (°C)")
    ax_bias.set_title("TCRE bias")
    ax_bias.legend(fontsize=8)
    ax_bias.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")

    # ── Structured summary: per-scenario + combined slopes, for training log ──
    def _standalone_regression(sc, key_ens, key_years):
        d = results.get(sc)
        if d is None or d.get(key_ens) is None:
            return None, None, 0
        yr = d[key_years]
        en = d[key_ens].mean(axis=0)
        lookup_key = sc if sc in lookups else (PROJECTIONS[0] if PROJECTIONS else None)
        if lookup_key is None:
            return None, None, 0
        cc = get_cumco2(yr, lookup_key)
        v  = ~np.isnan(cc)
        if v.sum() < 5:
            return None, None, 0
        a, b = np.polyfit(cc[v], en[v], 1)
        return float(a), float(b), int(v.sum())

    # Build a standalone ghg cumulative-CO2 lookup if ghg data are present,
    # so its TCRE slope is computed against its own cumCO2 trajectory.
    if "ghg" in results and results["ghg"].get("co2_annual") is not None:
        d = results["ghg"]
        gy = np.asarray(d["co2_years"]).astype(int)
        gc = np.asarray(d["co2_annual"])
        order = np.argsort(gy)
        gy, gc = gy[order], gc[order]
        lookups["ghg"] = dict(zip(gy, np.cumsum(gc)))

    summary = {"per_scenario": {}, "combined": {}}
    for sc in ("hist", "ssp370", "ssp126", "ghg"):
        if sc not in results:
            continue
        ms, _, n_m = _standalone_regression(sc, "gen_anom_ens", "gen_years")
        cs, _, n_c = _standalone_regression(sc, "cesm_anom_ens", "cesm_years")
        if ms is None or cs is None or cs == 0:
            continue
        summary["per_scenario"][sc] = {
            "model_slope": ms,
            "cesm_slope":  cs,
            "ratio":       ms / cs,
            "bias_pct":    100.0 * (ms / cs - 1.0),
            "n_points":    max(n_m, n_c),
        }
    for proj in PROJECTIONS:
        ms, _ = _combined_regression(proj, "gen_anom_ens", "gen_years")
        cs, _ = _combined_regression(proj, "cesm_anom_ens", "cesm_years")
        if ms is None or cs is None or cs == 0:
            continue
        summary["combined"][f"hist+{proj}"] = {
            "model_slope": float(ms),
            "cesm_slope":  float(cs),
            "ratio":       float(ms / cs),
            "bias_pct":    float(100.0 * (ms / cs - 1.0)),
        }

    # Grep-friendly block for log scraping by the trainer
    print("[TCRE SUMMARY]")
    for sc, s in summary["per_scenario"].items():
        print(f"  [TCRE] {sc:12s} model={s['model_slope']:.4f}  "
              f"CESM2={s['cesm_slope']:.4f}  ratio={s['ratio']:.3f}  "
              f"bias={s['bias_pct']:+.1f}%  N={s['n_points']}")
    for sc, s in summary["combined"].items():
        print(f"  [TCRE] {sc:12s} model={s['model_slope']:.4f}  "
              f"CESM2={s['cesm_slope']:.4f}  ratio={s['ratio']:.3f}  "
              f"bias={s['bias_pct']:+.1f}%")

    try:
        import json
        summary_path = os.path.join(os.path.dirname(out_path), "tcre_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  → saved {summary_path}")
    except Exception as e:
        print(f"  [TCRE] WARNING: failed to write tcre_summary.json: {e}")


def compute_ig_per_output_location(
    model: UNetModel3D,
    scheduler: ContinuousDDPM,
    cond_tensor: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    years: np.ndarray,
    windows: list,
    output_locations: list = OUTPUT_LOCATIONS,
    n_ig_steps: int = 20,
    batch_size: int = 4,
    t_proxy: float = 0.2,
    seed: int = 42,
) -> dict:
    """Compute Integrated Gradients of T at each output location w.r.t. conditioning.

    For each output location (lat_o, lon_o), computes IG of the predicted
    temperature at that specific grid cell w.r.t. CO2 and SUL conditioning maps.

    The resulting attribution map answers: "which conditioning grid points drive
    the temperature prediction at this output location?" — i.e., the model's
    implicit teleconnection from forcing to local temperature.

    One backward pass per output location per IG step, so cost scales as
    K × n_ig_steps × T rather than H×W × n_ig_steps × T.

    Returns
    -------
    dict[location_name][window_label] = {
        "co2": (H, W),    # mean |IG| from CO2 conditioning
        "sul": (H, W),    # mean |IG| from SUL conditioning
        "lat_idx": int,
        "lon_idx": int,
    }
    """
    _, T_total, H, W = cond_tensor.shape

    # Snap output locations to nearest grid cell
    loc_indices = []
    for loc_name, lat_o, lon_o in output_locations:
        lat_idx = int(np.argmin(np.abs(lat - lat_o)))
        lon_idx = int(np.argmin(np.abs(lon - lon_o)))
        loc_indices.append((loc_name, lat_idx, lon_idx))
        print(f"  [IG loc] {loc_name:12s}: grid ({lat_idx:3d},{lon_idx:3d})"
              f" = ({lat[lat_idx]:+6.1f}°N, {lon[lon_idx]:+7.1f}°E)")

    K = len(loc_indices)

    # Running (T, H, W) IG accumulator per location and channel
    ig_raw = {
        loc_name: {
            "co2": np.zeros((T_total, H, W), dtype=np.float32),
            "sul": np.zeros((T_total, H, W), dtype=np.float32),
        }
        for loc_name, _, _ in loc_indices
    }

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for t_start in tqdm(range(0, T_total, batch_size), desc="  IG batches"):
        t_end = min(t_start + batch_size, T_total)
        B = t_end - t_start

        cond_actual = (
            cond_tensor[:, t_start:t_end]    # (2, B, H, W)
            .permute(1, 0, 2, 3)             # (B, 2, H, W)
            .unsqueeze(2)                    # (B, 2, 1, H, W)
            .to(device=device, dtype=dtype)
        )
        cond_null_b = torch.full_like(cond_actual, NULL_COND)
        delta = cond_actual - cond_null_b    # (B, 2, 1, H, W)

        noise   = torch.randn(B, 1, 1, H, W, device=device, dtype=dtype, generator=rng)
        t_val   = torch.full((B,), t_proxy, device=device, dtype=dtype)
        log_snr = scheduler.log_snr(t_val)
        x_clean = torch.zeros(B, 1, 1, H, W, device=device, dtype=dtype)
        x_noisy = scheduler.add_noise(x_clean, noise, log_snr).detach()
        log_snr = log_snr.detach()

        # Gradient accumulators: (B, H, W) per location and channel
        sum_grads = {
            loc_name: {
                "co2": torch.zeros(B, H, W, device=device, dtype=dtype),
                "sul": torch.zeros(B, H, W, device=device, dtype=dtype),
            }
            for loc_name, _, _ in loc_indices
        }

        for k in range(1, n_ig_steps + 1):
            alpha = k / n_ig_steps
            cond_k = (cond_null_b + alpha * delta).detach().requires_grad_(True)

            v_pred  = model(x_noisy, log_snr, cond_map=cond_k)
            pred_x0 = scheduler.predict_start_from_v(x_noisy, log_snr, v_pred)
            pred_map = pred_x0.squeeze(1).squeeze(1)    # (B, H, W)

            # One backward per output location; retain graph for all but the last
            for i, (loc_name, lat_idx, lon_idx) in enumerate(loc_indices):
                T_local = pred_map[:, lat_idx, lon_idx]   # (B,)
                retain  = (i < K - 1)
                grads   = torch.autograd.grad(
                    T_local.sum(), cond_k, retain_graph=retain
                )[0]    # (B, 2, 1, H, W)
                sum_grads[loc_name]["co2"] += grads[:, 0, 0]
                sum_grads[loc_name]["sul"] += grads[:, 1, 0]

        delta_co2 = delta[:, 0, 0].detach()    # (B, H, W)
        delta_sul = delta[:, 1, 0].detach()

        for loc_name, _, _ in loc_indices:
            ig_raw[loc_name]["co2"][t_start:t_end] = (
                delta_co2 * (sum_grads[loc_name]["co2"] / n_ig_steps)
            ).cpu().numpy()
            ig_raw[loc_name]["sul"][t_start:t_end] = (
                delta_sul * (sum_grads[loc_name]["sul"] / n_ig_steps)
            ).cpu().numpy()

    # Average |IG| over time windows
    result = {}
    for loc_name, lat_idx, lon_idx in loc_indices:
        result[loc_name] = {}
        for y_start, y_end, label in windows:
            mask = (years >= y_start) & (years <= y_end)
            if not mask.any():
                continue
            result[loc_name][label] = {
                "co2":     np.abs(ig_raw[loc_name]["co2"][mask]).mean(axis=0),
                "sul":     np.abs(ig_raw[loc_name]["sul"][mask]).mean(axis=0),
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
            }
    return result


def compute_saliency_per_output_location(
    model: UNetModel3D,
    scheduler: ContinuousDDPM,
    cond_tensor: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    years: np.ndarray,
    windows: list,
    output_locations: list = OUTPUT_LOCATIONS,
    batch_size: int = 4,
    t_proxy: float = 0.2,
    seed: int = 42,
) -> dict:
    """Compute saliency maps of T at each output location w.r.t. conditioning.

    Saliency = |∂T(lat_o, lon_o) / ∂cond| evaluated at the actual conditioning
    (no baseline interpolation). Much faster than IG — one forward + K backward
    passes per batch — but can saturate in flat regions of the input space.

    Returns same dict structure as compute_ig_per_output_location.
    """
    _, T_total, H, W = cond_tensor.shape

    loc_indices = []
    for loc_name, lat_o, lon_o in output_locations:
        lat_idx = int(np.argmin(np.abs(lat - lat_o)))
        lon_idx = int(np.argmin(np.abs(lon - lon_o)))
        loc_indices.append((loc_name, lat_idx, lon_idx))
        print(f"  [SAL loc] {loc_name:12s}: grid ({lat_idx:3d},{lon_idx:3d})"
              f" = ({lat[lat_idx]:+6.1f}°N, {lon[lon_idx]:+7.1f}°E)")

    K = len(loc_indices)

    sal_raw = {
        loc_name: {
            "co2": np.zeros((T_total, H, W), dtype=np.float32),
            "sul": np.zeros((T_total, H, W), dtype=np.float32),
        }
        for loc_name, _, _ in loc_indices
    }

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for t_start in tqdm(range(0, T_total, batch_size), desc="  Saliency batches"):
        t_end = min(t_start + batch_size, T_total)
        B = t_end - t_start

        cond_actual = (
            cond_tensor[:, t_start:t_end]
            .permute(1, 0, 2, 3)
            .unsqueeze(2)
            .to(device=device, dtype=dtype)
        ).detach().requires_grad_(True)

        noise   = torch.randn(B, 1, 1, H, W, device=device, dtype=dtype, generator=rng)
        t_val   = torch.full((B,), t_proxy, device=device, dtype=dtype)
        log_snr = scheduler.log_snr(t_val)
        x_clean = torch.zeros(B, 1, 1, H, W, device=device, dtype=dtype)
        x_noisy = scheduler.add_noise(x_clean, noise, log_snr).detach()
        log_snr = log_snr.detach()

        v_pred   = model(x_noisy, log_snr, cond_map=cond_actual)
        pred_x0  = scheduler.predict_start_from_v(x_noisy, log_snr, v_pred)
        pred_map = pred_x0.squeeze(1).squeeze(1)    # (B, H, W)

        for i, (loc_name, lat_idx, lon_idx) in enumerate(loc_indices):
            T_local = pred_map[:, lat_idx, lon_idx]   # (B,)
            retain  = (i < K - 1)
            grads   = torch.autograd.grad(
                T_local.sum(), cond_actual, retain_graph=retain
            )[0]    # (B, 2, 1, H, W)
            sal_raw[loc_name]["co2"][t_start:t_end] = grads[:, 0, 0].detach().abs().cpu().numpy()
            sal_raw[loc_name]["sul"][t_start:t_end] = grads[:, 1, 0].detach().abs().cpu().numpy()

    result = {}
    for loc_name, lat_idx, lon_idx in loc_indices:
        result[loc_name] = {}
        for y_start, y_end, label in windows:
            mask = (years >= y_start) & (years <= y_end)
            if not mask.any():
                continue
            result[loc_name][label] = {
                "co2":     sal_raw[loc_name]["co2"][mask].mean(axis=0),
                "sul":     sal_raw[loc_name]["sul"][mask].mean(axis=0),
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
            }
    return result


def compute_baseline(gmean: np.ndarray, years: np.ndarray,
                     start=BASELINE_START, end=BASELINE_END) -> float:
    mask = (years >= start) & (years <= end)
    if not mask.any():
        raise ValueError(f"No years in [{start},{end}] for baseline")
    return float(gmean[mask].mean())


# ─────────────────────────────────────────────────────────────────────────────
# NetCDF output
# ─────────────────────────────────────────────────────────────────────────────

def save_netcdf(
    name: str,
    gen_ensemble: np.ndarray,
    gen_years: np.ndarray,
    baseline_map: np.ndarray,
    cesm_ensemble: np.ndarray | None,
    cesm_years: np.ndarray | None,
    out_path: str,
    ckpt_path: str,
    gen_baseline_map: np.ndarray | None = None,
):
    """Save ensemble model output (and optionally CESM2 reference) to NetCDF.

    Variables written
    -----------------
    TREFHT_model_mean          (year, lat, lon)  — ensemble mean [°C]
    TREFHT_model_mean_anom     (year, lat, lon)  — ensemble mean anomaly [°C]
    TREFHT_model_gmean_mean    (year,)           — ensemble mean global-mean [°C]
    TREFHT_model_gmean_mean_anom (year,)         — ensemble mean global-mean anomaly [°C]
    TREFHT_model_mN            (year, lat, lon)  — member N absolute [°C]
    TREFHT_model_mN_anom       (year, lat, lon)  — member N anomaly [°C]
    TREFHT_model_gmean_mN      (year,)           — member N global-mean [°C]
    TREFHT_model_gmean_mN_anom (year,)           — member N global-mean anomaly [°C]
    baseline_map               (lat, lon)        — 1850-1900 CESM2 climatology [°C]
    TREFHT_model_baseline      (lat, lon)        — 1850-1900 model climatology [°C]

    If cesm_data is provided, also writes:
    TREFHT_cesm / _anom / _gmean / _gmean_anom
    """
    N_ENS = gen_ensemble.shape[0]
    gen_mean = gen_ensemble.mean(axis=0)           # (T, H, W) ensemble mean

    coords_model = {"year": gen_years, "lat": LAT, "lon": LON}
    w = np.cos(np.deg2rad(LAT))[:, np.newaxis]
    w = w / w.mean()
    bl_scalar = float((baseline_map * w).mean())

    anom_mean       = gen_mean - baseline_map
    gmean_mean      = (gen_mean * w).mean(axis=(-2, -1))
    gmean_mean_anom = gmean_mean - bl_scalar

    ds = xr.Dataset(
        {
            "TREFHT_model_mean": xr.DataArray(
                gen_mean, dims=["year", "lat", "lon"], coords=coords_model,
                attrs={"units": "degC", "long_name": f"Ensemble mean model TREFHT (N={N_ENS})"}),
            "TREFHT_model_mean_anom": xr.DataArray(
                anom_mean, dims=["year", "lat", "lon"], coords=coords_model,
                attrs={"units": "degC", "long_name": "Ensemble mean TREFHT anomaly re 1850-1900"}),
            "TREFHT_model_gmean_mean": xr.DataArray(
                gmean_mean, dims=["year"], coords={"year": gen_years},
                attrs={"units": "degC", "long_name": "Ensemble mean global-mean TREFHT"}),
            "TREFHT_model_gmean_mean_anom": xr.DataArray(
                gmean_mean_anom, dims=["year"], coords={"year": gen_years},
                attrs={"units": "degC", "long_name": "Ensemble mean global-mean TREFHT anomaly re 1850-1900"}),
            "baseline_map": xr.DataArray(
                baseline_map, dims=["lat", "lon"], coords={"lat": LAT, "lon": LON},
                attrs={"units": "degC", "long_name": "1850-1900 climatological mean (CESM2)"}),
        },
        attrs={
            "experiment":        name,
            "checkpoint":        os.path.basename(ckpt_path),
            "baseline":          f"{BASELINE_START}-{BASELINE_END}",
            "n_model_ensemble":  N_ENS,
            "description":       "CESM2 aerosol emulator evaluation output",
        },
    )

    # ── per-member variables ──────────────────────────────────────────────────
    for m in range(N_ENS):
        mem = gen_ensemble[m]                          # (T, H, W)
        anom_m      = mem - baseline_map
        gmean_m     = (mem * w).mean(axis=(-2, -1))
        gmean_m_anom = gmean_m - bl_scalar
        tag = f"m{m + 1}"
        ds[f"TREFHT_model_{tag}"] = xr.DataArray(
            mem, dims=["year", "lat", "lon"], coords=coords_model,
            attrs={"units": "degC", "long_name": f"Model TREFHT member {m + 1}"})
        ds[f"TREFHT_model_{tag}_anom"] = xr.DataArray(
            anom_m, dims=["year", "lat", "lon"], coords=coords_model,
            attrs={"units": "degC", "long_name": f"Model TREFHT anomaly member {m + 1}"})
        ds[f"TREFHT_model_gmean_{tag}"] = xr.DataArray(
            gmean_m, dims=["year"], coords={"year": gen_years},
            attrs={"units": "degC", "long_name": f"Global-mean TREFHT member {m + 1}"})
        ds[f"TREFHT_model_gmean_{tag}_anom"] = xr.DataArray(
            gmean_m_anom, dims=["year"], coords={"year": gen_years},
            attrs={"units": "degC", "long_name": f"Global-mean TREFHT anomaly member {m + 1}"})

    if gen_baseline_map is not None:
        ds["TREFHT_model_baseline"] = xr.DataArray(
            gen_baseline_map, dims=["lat", "lon"], coords={"lat": LAT, "lon": LON},
            attrs={"units": "degC",
                   "long_name": f"Model {BASELINE_START}-{BASELINE_END} climatological mean"})

    if cesm_ensemble is not None and cesm_years is not None:
        N_CESM = cesm_ensemble.shape[0]
        cesm_data = cesm_ensemble.mean(axis=0)          # (T, H, W) ensemble mean
        coords_cesm = {"cesm_year": cesm_years, "lat": LAT, "lon": LON}
        gmean_cesm      = (cesm_data * w).mean(axis=(-2, -1))
        anom_cesm       = cesm_data - baseline_map
        gmean_cesm_anom = gmean_cesm - bl_scalar
        ds["TREFHT_cesm_mean"] = xr.DataArray(
            cesm_data, dims=["cesm_year", "lat", "lon"], coords=coords_cesm,
            attrs={"units": "degC", "long_name": f"CESM2 TREFHT ensemble mean (N={N_CESM})"})
        ds["TREFHT_cesm_mean_anom"] = xr.DataArray(
            anom_cesm, dims=["cesm_year", "lat", "lon"], coords=coords_cesm,
            attrs={"units": "degC", "long_name": "CESM2 ensemble mean TREFHT anomaly re 1850-1900"})
        ds["TREFHT_cesm_gmean_mean"] = xr.DataArray(
            gmean_cesm, dims=["cesm_year"], coords={"cesm_year": cesm_years},
            attrs={"units": "degC", "long_name": "CESM2 ensemble mean global-mean TREFHT"})
        ds["TREFHT_cesm_gmean_mean_anom"] = xr.DataArray(
            gmean_cesm_anom, dims=["cesm_year"], coords={"cesm_year": cesm_years},
            attrs={"units": "degC", "long_name": "CESM2 ensemble mean global-mean TREFHT anomaly re 1850-1900"})
        # per-member CESM2 variables
        for m in range(N_CESM):
            mem = cesm_ensemble[m]
            anom_m = mem - baseline_map
            gmean_m = (mem * w).mean(axis=(-2, -1))
            gmean_m_anom = gmean_m - bl_scalar
            tag = f"m{m + 1}"
            ds[f"TREFHT_cesm_{tag}"] = xr.DataArray(
                mem, dims=["cesm_year", "lat", "lon"], coords=coords_cesm,
                attrs={"units": "degC", "long_name": f"CESM2 TREFHT member {m + 1}"})
            ds[f"TREFHT_cesm_{tag}_anom"] = xr.DataArray(
                anom_m, dims=["cesm_year", "lat", "lon"], coords=coords_cesm,
                attrs={"units": "degC", "long_name": f"CESM2 TREFHT anomaly member {m + 1}"})
            ds[f"TREFHT_cesm_gmean_{tag}"] = xr.DataArray(
                gmean_m, dims=["cesm_year"], coords={"cesm_year": cesm_years},
                attrs={"units": "degC", "long_name": f"CESM2 global-mean TREFHT member {m + 1}"})
            ds[f"TREFHT_cesm_gmean_{tag}_anom"] = xr.DataArray(
                gmean_m_anom, dims=["cesm_year"], coords={"cesm_year": cesm_years},
                attrs={"units": "degC", "long_name": f"CESM2 global-mean TREFHT anomaly member {m + 1}"})

    ds.to_netcdf(out_path)
    print(f"  → saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(results: dict, out_path: str):
    """results[name] = dict(gen_anom, cesm_anom, gen_years, cesm_years, color)

    Top panel : anomaly time series — model (solid) vs CESM2 member (dashed)
    Bottom panel : bias = model − CESM2 on common years
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for name, d in results.items():
        c = d["color"]
        gen_anom_ens  = d["gen_anom_ens"]            # (N_ENS, T)
        gen_anom_mean = gen_anom_ens.mean(axis=0)    # (T,)
        gen_years     = d["gen_years"]
        N_gen = gen_anom_ens.shape[0]

        if N_gen == 1:
            # Single member — just draw the line, no spread
            ax_top.plot(gen_years, gen_anom_mean, color=c, lw=2.0,
                        label=f"{name} (model)")
        else:
            # Multiple members — shaded min/max spread + mean
            gen_lo = gen_anom_ens.min(axis=0)
            gen_hi = gen_anom_ens.max(axis=0)
            ax_top.fill_between(gen_years, gen_lo, gen_hi, color=c, alpha=0.18)
            ax_top.plot(gen_years, gen_anom_mean, color=c, lw=2.0,
                        label=f"{name} (model mean ± spread, N={N_gen})")

        if d.get("cesm_anom") is not None:
            cesm_anom_ens  = d["cesm_anom_ens"]    # (N_CESM, T)
            cesm_anom_mean = d["cesm_anom"]          # (T,)
            cesm_years     = d["cesm_years"]
            N_cesm = cesm_anom_ens.shape[0]

            if N_cesm == 1:
                ax_top.plot(cesm_years, cesm_anom_mean, color=c, lw=2.0,
                            ls="--", alpha=0.8, label=f"{name} (CESM2)")
            else:
                cesm_lo = cesm_anom_ens.min(axis=0)
                cesm_hi = cesm_anom_ens.max(axis=0)
                ax_top.fill_between(cesm_years, cesm_lo, cesm_hi,
                                    facecolor=c, alpha=0.10, lw=0)
                ax_top.fill_between(cesm_years, cesm_lo, cesm_hi,
                                    facecolor="none", hatch="///",
                                    edgecolor=c, alpha=0.4, lw=0)
                ax_top.plot(cesm_years, cesm_anom_mean, color=c, lw=2.0,
                            ls="--", alpha=0.8,
                            label=f"{name} (CESM2 mean ± spread, N={N_cesm})")

            common, idx_gen, idx_cs = np.intersect1d(
                gen_years, cesm_years, return_indices=True
            )
            # bias: model ensemble mean vs CESM2 ensemble mean
            diff_mean = gen_anom_mean[idx_gen] - cesm_anom_mean[idx_cs]
            ax_bot.plot(common, diff_mean, color=c, lw=1.5, label=name)
            # shade model spread around bias
            if N_gen > 1:
                diff_lo = gen_anom_ens[:, idx_gen].min(axis=0) - cesm_anom_mean[idx_cs]
                diff_hi = gen_anom_ens[:, idx_gen].max(axis=0) - cesm_anom_mean[idx_cs]
                ax_bot.fill_between(common, diff_lo, diff_hi, alpha=0.15, color=c)

    # grey ±|member diff| band on bias panel (internal variability reference)
    ref_band_drawn = False
    for name, d in results.items():
        if d.get("ref_diff") is not None and d.get("ref_years") is not None:
            rd = np.abs(d["ref_diff"])
            ry = d["ref_years"]
            label = "±|member diff| (nat. var.)" if not ref_band_drawn else None
            ax_bot.fill_between(ry, -rd, rd, color="grey", alpha=0.20,
                                zorder=0, label=label)
            ref_band_drawn = True

    n_gen_label = gen_anom_ens.shape[0] if results else 1
    ax_top.axhline(0, color="k", lw=0.6, ls=":")
    ax_top.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12, label="baseline period")
    ax_top.set_xlabel("")
    ax_top.set_ylabel("TREFHT anomaly (°C)")
    ax_top.set_title(
        f"Global-mean temperature anomaly vs 1850–1900  "
        f"(model solid, CESM2 dashed — {n_gen_label}-member ensemble)"
    )
    ax_top.legend(fontsize=8, ncol=2)
    ax_top.grid(True, alpha=0.25)

    ax_bot.axhline(0, color="k", lw=0.9)
    ax_bot.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12)
    ax_bot.set_xlabel("Year")
    ax_bot.set_ylabel("Bias: model − CESM2 (°C)")
    ax_bot.set_title(
        "Model bias relative to CESM2 ensemble mean"
        + (" (shaded = model min/max spread)" if n_gen_label > 1 else "")
    )
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def save_csv(results: dict, out_path: str):
    """Save global-mean anomaly and bias to a CSV file.

    Columns: experiment, year, model_anom_degC, cesm_anom_degC, bias_degC
    bias = model_anom - cesm_anom on common years; NaN where CESM2 unavailable.
    """
    import csv
    rows = []
    for name, d in results.items():
        gen_anom_mean = d["gen_anom_ens"].mean(axis=0)   # (T,)
        gen_years     = d["gen_years"]

        if d.get("cesm_anom") is not None:
            cesm_anom_mean = d["cesm_anom"]
            cesm_years     = d["cesm_years"]
            common, idx_gen, idx_cs = np.intersect1d(
                gen_years, cesm_years, return_indices=True
            )
            cesm_lookup = {int(yr): float(cesm_anom_mean[i])
                           for yr, i in zip(common, idx_cs)}
        else:
            cesm_lookup = {}

        for i, yr in enumerate(gen_years):
            yr = int(yr)
            model_anom = float(gen_anom_mean[i])
            cesm_anom  = cesm_lookup.get(yr, float("nan"))
            bias       = model_anom - cesm_anom if not np.isnan(cesm_anom) else float("nan")
            rows.append({
                "experiment":     name,
                "year":           yr,
                "model_anom_degC": round(model_anom, 4),
                "cesm_anom_degC":  round(cesm_anom, 4) if not np.isnan(cesm_anom) else "",
                "bias_degC":       round(bias, 4)       if not np.isnan(bias)      else "",
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "year", "model_anom_degC",
                           "cesm_anom_degC", "bias_degC"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → saved {out_path}")


def _nearest_year(years: np.ndarray, target: int) -> int:
    """Return the year in `years` closest to `target`."""
    idx = np.argmin(np.abs(years - target))
    return int(years[idx])


def _window_indices(years: np.ndarray, target: int, window: int = 10):
    """Indices of years inside a `window`-year span centered on `target`.
    For even windows, span is [target - window//2 + 1, target + window//2]
    (clipped to the available range)."""
    half = window // 2
    lo = target - half + 1
    hi = target + half
    mask = (years >= lo) & (years <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(np.abs(years - target)))])
    return idx


def plot_anomaly_maps(name: str, gen_data: np.ndarray, gen_years: np.ndarray,
                      baseline_map: np.ndarray, map_years: list,
                      cesm_data: np.ndarray | None, cesm_years: np.ndarray | None,
                      out_path: str,
                      gen_ensemble: np.ndarray | None = None,
                      cesm_ensemble: np.ndarray | None = None):
    """Spatial anomaly maps at requested years.

    gen_data    : (T, H, W)       generated temperature ensemble mean [°C]
    baseline_map: (H, W)          time-mean over 1850-1900 from hist
    gen_ensemble : (N, T, H, W)   individual model members (optional)
    cesm_ensemble: (M, T, H, W)   individual CESM2 members (optional)

    Rows:
      0 — Model anomaly  (re 1850-1900)
      1 — CESM2 anomaly  (re 1850-1900)   [only if cesm_data provided]
      2 — Difference: Model − CESM2        [only if cesm_data provided]
          Stippling marks grid cells where the difference is statistically
          significant (Welch t-test p < 0.05 across ensemble members),
          i.e. not explained by natural climate variability.
    """
    has_cesm = (cesm_data is not None) and (cesm_years is not None)
    n_cols = len(map_years)
    n_rows = 3 if has_cesm else 1
    row_labels = ["Model"]
    if has_cesm:
        row_labels += ["CESM2", "Model − CESM2"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    vmax_anom = 4.0
    vmax_diff = 2.0
    cmap = plt.cm.RdBu_r
    norm_anom = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax_anom, vmax=vmax_anom)
    norm_diff = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax_diff, vmax=vmax_diff)

    def _plot_panel(ax, data, norm, title):
        data_cyc, lon_cyc = add_cyclic_point(data, coord=LON)
        da = xr.DataArray(
            data_cyc, dims=["lat", "lon"],
            coords={"lat": LAT, "lon": lon_cyc},
            attrs={"units": "°C"},
        )
        da.plot.pcolormesh(
            ax=ax, cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            add_colorbar=True,
            cbar_kwargs={"label": "°C", "shrink": 0.75},
        )
        ax.add_feature(cfeature.COASTLINE, lw=0.5)
        ax.add_feature(cfeature.BORDERS, lw=0.3, linestyle=":")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                          color="grey", alpha=0.5, linestyle="--")
        gl.top_labels   = False
        gl.right_labels = False
        ax.set_title(title, fontsize=9)
        # Global mean annotation in bottom-right corner
        gmean = float(area_weighted_gmean(data[np.newaxis], LAT)[0])
        ax.text(0.98, 0.03, f"GM: {gmean:+.2f}°C",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    win = 10
    for col, yr_target in enumerate(map_years):
        idx_gen = _window_indices(gen_years, yr_target, win)
        win_gen = (int(gen_years[idx_gen[0]]), int(gen_years[idx_gen[-1]]))
        anom_gen = gen_data[idx_gen].mean(axis=0) - baseline_map  # (H, W)

        _plot_panel(axes[0, col], anom_gen, norm_anom,
                    f"{name} model  ({win_gen[0]}–{win_gen[1]})")

        if has_cesm:
            idx_cs = _window_indices(cesm_years, yr_target, win)
            win_cs = (int(cesm_years[idx_cs[0]]), int(cesm_years[idx_cs[-1]]))
            anom_cs = cesm_data[idx_cs].mean(axis=0) - baseline_map

            _plot_panel(axes[1, col], anom_cs, norm_anom,
                        f"{name} CESM2  ({win_cs[0]}–{win_cs[1]})")
            _plot_panel(axes[2, col], anom_gen - anom_cs, norm_diff,
                        f"Model − CESM2  ({win_gen[0]}–{win_gen[1]})")

            # Stipple where difference is significant vs natural variability
            if gen_ensemble is not None and cesm_ensemble is not None:
                # Per-member window-mean anomalies: (N, H, W) and (M, H, W)
                gen_members  = gen_ensemble[:, idx_gen].mean(axis=1)  - baseline_map
                cesm_members = cesm_ensemble[:, idx_cs].mean(axis=1) - baseline_map
                # Welch's t-test at each grid point
                _, pvals = stats.ttest_ind(gen_members, cesm_members,
                                           axis=0, equal_var=False)       # (H, W)
                sig_mask = pvals < 0.05
                # Overlay stipple dots on significant grid cells
                lon_idx, lat_idx = np.meshgrid(np.arange(sig_mask.shape[1]),
                                               np.arange(sig_mask.shape[0]))
                sig_lats = LAT[lat_idx[sig_mask]]
                sig_lons = LON[lon_idx[sig_mask]]
                ax_diff = axes[2, col]
                plot_kw = dict(color="k", s=0.3, alpha=0.5, linewidths=0, rasterized=True)
                ax_diff.scatter(sig_lons, sig_lats, transform=ccrs.PlateCarree(),
                                zorder=5, **plot_kw)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f"TREFHT anomaly vs 1850–1900 — {name} (10-yr mean centered on target)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def _plot_loc_attribution(name: str, results: dict, out_path_prefix: str,
                          method: str, cbar_label: str):
    """Shared plot helper for per-location attribution maps (IG or saliency).

    One figure per output location saved as {out_path_prefix}_{loc_name}.png.
    Rows: CO2, SUL, raw difference (|CO2|-|SUL|), normalised ratio.
    Columns: one per time window.  ★ marks the output location on every panel.
    """
    if not results:
        print(f"  [{method}] No results to plot for {name}, skipping.")
        return

    for loc_name, window_data in results.items():
        windows = list(window_data.keys())
        if not windows:
            continue

        n_cols = len(windows)
        n_rows = 4   # CO2 | SUL | raw diff | normalised ratio

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 3.5 * n_rows),
            subplot_kw={"projection": ccrs.PlateCarree()},
            squeeze=False,
        )

        # Shared colour scale per channel (98th percentile across all windows)
        vmax_co2 = max(
            np.percentile(window_data[w]["co2"], 98) for w in windows
        )
        vmax_sul = max(
            np.percentile(window_data[w]["sul"], 98) for w in windows
        )
        vmax_co2 = max(vmax_co2, 1e-9)
        vmax_sul = max(vmax_sul, 1e-9)

        # Shared symmetric colour scale for raw difference
        diffs = [window_data[w]["co2"] - window_data[w]["sul"] for w in windows]
        vmax_diff = max(np.percentile(np.abs(d), 98) for d in diffs)
        vmax_diff = max(vmax_diff, 1e-9)

        lat_idx = window_data[windows[0]]["lat_idx"]
        lon_idx = window_data[windows[0]]["lon_idx"]
        out_lat = float(LAT[lat_idx])
        out_lon = float(LON[lon_idx])

        def _star(ax):
            ax.plot(out_lon, out_lat, transform=ccrs.PlateCarree(),
                    marker="*", color="blue", markersize=12,
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=10, linestyle="none")

        def _gridlines(ax):
            gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                              color="grey", alpha=0.5, linestyle="--")
            gl.top_labels   = False
            gl.right_labels = False

        def _plot_map(ax, data, vmax, title):
            data_cyc, lon_cyc = add_cyclic_point(data, coord=LON)
            da = xr.DataArray(data_cyc, dims=["lat", "lon"],
                              coords={"lat": LAT, "lon": lon_cyc})
            da.plot.pcolormesh(ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax,
                               transform=ccrs.PlateCarree(), add_colorbar=True,
                               cbar_kwargs={"label": cbar_label, "shrink": 0.75})
            ax.add_feature(cfeature.COASTLINE, lw=0.5)
            ax.add_feature(cfeature.BORDERS, lw=0.3, linestyle=":")
            _gridlines(ax)
            ax.set_title(title, fontsize=9)
            _star(ax)

        def _plot_div(ax, data, vmax, title, cbar_lbl):
            data_cyc, lon_cyc = add_cyclic_point(data, coord=LON)
            da = xr.DataArray(data_cyc, dims=["lat", "lon"],
                              coords={"lat": LAT, "lon": lon_cyc})
            da.plot.pcolormesh(ax=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               transform=ccrs.PlateCarree(), add_colorbar=True,
                               cbar_kwargs={"label": cbar_lbl, "shrink": 0.75})
            ax.add_feature(cfeature.COASTLINE, lw=0.5)
            ax.add_feature(cfeature.BORDERS, lw=0.3, linestyle=":")
            _gridlines(ax)
            ax.set_title(title, fontsize=9)
            _star(ax)

        for col, window in enumerate(windows):
            co2 = window_data[window]["co2"]
            sul = window_data[window]["sul"]
            raw_diff  = co2 - sul
            denom     = co2 + sul
            norm_ratio = np.where(denom > 1e-12, raw_diff / denom, 0.0)

            _plot_map(axes[0, col], co2, vmax_co2,
                      f"CO2 → {loc_name}\n{window}")
            _plot_map(axes[1, col], sul, vmax_sul,
                      f"SUL → {loc_name}\n{window}")
            _plot_div(axes[2, col], raw_diff, vmax_diff,
                      f"|CO2|−|SUL| → {loc_name}\n{window}",
                      f"Δ{cbar_label}  (red=CO2 dom.)")
            _plot_div(axes[3, col], norm_ratio, 1.0,
                      f"(|CO2|−|SUL|)/(|CO2|+|SUL|) → {loc_name}\n{window}",
                      "ratio  (red=CO2, blue=SUL)")

        axes[0, 0].set_ylabel("CO2",         fontsize=10)
        axes[1, 0].set_ylabel("SUL",         fontsize=10)
        axes[2, 0].set_ylabel("Raw diff",    fontsize=10)
        axes[3, 0].set_ylabel("Norm. ratio", fontsize=10)

        fig.suptitle(
            f"{method} → T at {loc_name} ({out_lat:+.1f}°N, {out_lon:.1f}°E)"
            f" — {name}\n"
            "(★ = output location; red = CO2 dominates, blue = SUL dominates)",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = f"{out_path_prefix}_{loc_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  → saved {out_path}")


def plot_ig_per_location(name: str, ig_results: dict, out_path_prefix: str):
    """Plot IG attribution maps per output location."""
    _plot_loc_attribution(name, ig_results, out_path_prefix,
                          method="Integrated Gradients", cbar_label="|IG attr.|")


def plot_saliency_per_location(name: str, sal_results: dict, out_path_prefix: str):
    """Plot saliency maps per output location."""
    _plot_loc_attribution(name, sal_results, out_path_prefix,
                          method="Saliency", cbar_label="|∂T/∂cond|")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",    default=RUNS_DIR)
    parser.add_argument("--checkpoint",  default=None,
                        help="Path to a specific checkpoint file. "
                             "Overrides --runs-dir / find_latest_checkpoint.")
    parser.add_argument("--output-dir",  default="/scratch/project_462001328/eval_output")
    parser.add_argument("--sample-steps",  type=int, default=SAMPLE_STEPS)
    parser.add_argument("--batch-size",    type=int, default=BATCH_SIZE)
    parser.add_argument("--ig-n-steps",    type=int, default=30,
                        help="IG interpolation steps (more = more accurate, slower)")
    parser.add_argument("--ig-batch-size", type=int, default=8,
                        help="Years per GPU batch for IG (keep small to avoid OOM)")
    parser.add_argument("--skip-ig",            action="store_true", default=True,
                        help="Skip spatial IG attribution maps (saves time/memory; default: True)")
    parser.add_argument("--skip-saliency",      action="store_true", default=True,
                        help="Skip saliency maps (saves time/memory; default: True)")
    parser.add_argument("--run-xai",            action="store_true",
                        help="Run all XAI figures (IG + saliency); off by default")
    parser.add_argument("--saliency-batch-size", type=int, default=8,
                        help="Years per GPU batch for saliency (default: 8)")
    parser.add_argument("--guidance-co2", type=float, default=GUIDANCE_CO2,
                        help="Per-channel CFG scale for CO2 (default: %(default)s). "
                             "< 1.0 reduces CO2 warming contribution; enables 3-pass CFG.")
    parser.add_argument("--guidance-sul", type=float, default=GUIDANCE_SUL,
                        help="Per-channel CFG scale for SUL (default: %(default)s). "
                             "> 1.0 amplifies aerosol cooling; enables 3-pass CFG.")
    parser.add_argument("--force-cfg", action="store_true",
                        help="Always use the 3-pass CFG decomposition even when both "
                             "guidance scales are 1.0.  Useful to isolate the bias from "
                             "the additive decomposition vs the guidance scale values.")
    parser.add_argument("--members", type=int, default=N_ENSEMBLE,
                        help="Number of diffusion ensemble members to generate per "
                             "experiment (default: %(default)s).  More members give "
                             "a better estimate of model spread but take longer.")
    parser.add_argument("--export", default=None,
                        help="SLURM-style key=value pairs, e.g. 'members=10'.  "
                             "Parsed for 'members=N'; --members takes precedence.")
    parser.add_argument("--experiments", nargs="+", default=None,
                        metavar="NAME",
                        help="Only run these experiments by name (e.g. ssp126). "
                             "Default: all of "
                             f"{[e['name'] for e in EXPERIMENTS]}. "
                             "Baseline is still taken from hist CESM2 ensemble.")
    parser.add_argument("--fp32", action="store_true",
                        help="Force float32 inference (default: bf16). "
                             "Use to A/B against bf16 if precision is suspect.")
    parser.add_argument("--shard-rank", type=int,
                        default=int(os.environ.get("SLURM_PROCID", 0)),
                        help="Index of this shard among --n-shards. "
                             "Defaults to $SLURM_PROCID so srun --ntasks=N gives "
                             "automatic experiment-level parallelism.")
    parser.add_argument("--n-shards",   type=int,
                        default=int(os.environ.get("SLURM_NTASKS", 1)),
                        help="Total number of shards. Each shard runs the "
                             "subset experiments_to_run[rank::n_shards]. "
                             "Defaults to $SLURM_NTASKS.")
    args = parser.parse_args()

    # Parse --export="members=N" if --members was not set explicitly
    if args.export:
        for token in args.export.split(","):
            token = token.strip()
            if token.startswith("members="):
                try:
                    args.members = int(token.split("=", 1)[1])
                except ValueError:
                    pass
    if args.run_xai:
        args.skip_ig       = False
        args.skip_saliency = False

    os.makedirs(args.output_dir, exist_ok=True)

    # Sharded LUMI runs override ROCR_VISIBLE_DEVICES per rank, which can race
    # with SLURM's own GPU binding and leave torch.cuda.is_available()==False
    # on some ranks (e.g. ranks 0/1/3 in job 18580141 silently fell back to CPU
    # → Conv3D still ran but bf16 autocast on CPU produced abnormal outputs
    # that broke downstream save).  Fail loudly so the eval restarts under a
    # correct binding instead of producing partial output.
    if not torch.cuda.is_available():
        rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "<unset>")
        hipv = os.environ.get("HIP_VISIBLE_DEVICES",  "<unset>")
        procid = os.environ.get("SLURM_PROCID",  "<unset>")
        localid = os.environ.get("SLURM_LOCALID", "<unset>")
        sys.exit(f"[FATAL] torch.cuda.is_available()=False — refusing to run "
                 f"eval on CPU.  SLURM_PROCID={procid}  SLURM_LOCALID={localid}  "
                 f"ROCR_VISIBLE_DEVICES={rocr}  HIP_VISIBLE_DEVICES={hipv}")
    device = torch.device("cuda")
    # Mixed-precision inference via torch.autocast: model + tensors stay fp32,
    # individual ops are auto-cast to bf16 where supported and fall back to
    # fp32 where MIOpen lacks a bf16 kernel (e.g. some Conv3d shapes on
    # MI250x).  Casting the *whole* model to bf16 instead triggers a hard
    # dtype-mismatch crash when MIOpen falls back silently.
    # --fp32 disables autocast entirely for A/B testing.
    dtype       = torch.float32
    autocast_dt = None if args.fp32 else torch.bfloat16
    print(f"[DEVICE] {device}  dtype={dtype}  autocast={autocast_dt}")

    # ── load model ─────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint if args.checkpoint else find_latest_checkpoint(args.runs_dir)
    model, pca_state = load_model(ckpt_path, CONFIG_PATH, device)
    model = model.to(dtype)
    print(f"[PCA] {'Found in checkpoint' if pca_state else 'None — no PCA projection'}")

    pca_cond   = pca_state.get("cond")   if pca_state else None
    pca_target = pca_state.get("target") if pca_state else None

    # Read n_components_cond from config_data.yaml so eval uses the same
    # number of EOFs the model was trained with (currently [30, 10]).
    cfg = OmegaConf.load(CONFIG_PATH)
    data_cfg = OmegaConf.load("configs/config_data.yaml")
    _nc = data_cfg.get("n_components_cond", None)
    N_COMP_COND = OmegaConf.to_container(_nc, resolve=True) if (pca_cond and _nc is not None) else None
    print(f"[PCA] n_components_cond={N_COMP_COND}")

    # Mirror the training-side cond smoothing (data/climate_dataset.py). PCA is
    # loaded from the checkpoint and is currently absent, but the Gaussian
    # smoothing is deterministic and reproducible here from config alone — it
    # removes the inventory texture the model would otherwise imprint.
    _cs = data_cfg.get("cond_smooth_sigma", None)
    COND_SMOOTH_SIGMA = OmegaConf.to_container(_cs, resolve=True) if _cs is not None else None
    print(f"[COND] cond_smooth_sigma={COND_SMOOTH_SIGMA}")
    scheduler: ContinuousDDPM = instantiate(cfg.scheduler)

    # ── compute hist baseline map (H, W) for anomaly reference ─────────────
    print("\n[BASELINE] Loading hist CESM2 ensemble to compute 1850–1900 mean …")
    hist_exp = next(e for e in EXPERIMENTS if e["name"] == "hist")
    try:
        cesm_hist_years, cesm_hist_ens = load_cesm2_ensemble(
            hist_exp["data_dir"], hist_exp["realizations"], hist_exp["time_dim"],
            hist_exp.get("target_var", TARGET_VAR),
        )
        cesm_hist_data = cesm_hist_ens.mean(axis=0)   # ensemble mean (T, H, W)
        mask_bl = (cesm_hist_years >= BASELINE_START) & (cesm_hist_years <= BASELINE_END)
        baseline_map = cesm_hist_data[mask_bl].mean(axis=0)       # (H, W)  in °C
        print(f"  baseline map  mean={baseline_map.mean():.2f}°C  std={baseline_map.std():.2f}°C"
              f"  (from {cesm_hist_ens.shape[0]} members)")
    except Exception as exc:
        print(f"  WARNING: could not load CESM2 hist data ({exc})")
        print("  Will use model-generated hist 1850–1900 mean as baseline instead.")
        baseline_map = None   # computed later from generated hist

    # ── loop over experiments ───────────────────────────────────────────────
    timeseries_results = {}

    if args.experiments:
        wanted = set(args.experiments)
        unknown = wanted - {e["name"] for e in EXPERIMENTS}
        if unknown:
            sys.exit(f"ERROR: unknown experiment name(s): {sorted(unknown)}. "
                     f"Known: {[e['name'] for e in EXPERIMENTS]}")
        experiments_to_run = [e for e in EXPERIMENTS if e["name"] in wanted]
        print(f"\n[FILTER] Running only: {[e['name'] for e in experiments_to_run]}")
    else:
        experiments_to_run = EXPERIMENTS

    # ── Shard experiments across srun tasks for multi-GPU eval ───────────────
    # Each task takes experiments_to_run[rank::n_shards].  When n_shards==1
    # this is a no-op.  All shards still compute the hist CESM2 baseline
    # independently above, which is fine — it's cheap and keeps the shards
    # fully independent (no inter-rank IPC).
    if args.n_shards > 1:
        all_names = [e["name"] for e in experiments_to_run]
        experiments_to_run = experiments_to_run[args.shard_rank :: args.n_shards]
        print(f"[SHARD] rank={args.shard_rank}/{args.n_shards} "
              f"running={[e['name'] for e in experiments_to_run]} "
              f"(of {all_names})")

    for exp in experiments_to_run:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"[EXP] {name}")

        # -- conditioning --------------------------------------------------
        print("  Building conditioning tensor …")
        try:
            cond_tensor, cond_years, lat_file, lon_file = build_cond_tensor(
                exp["cond_file"], COND_VARS, exp["time_dim"],
                pca_cond, N_COMP_COND, COND_SMOOTH_SIGMA,
            )
        except Exception as e:
            print(f"  SKIP (conditioning failed): {e}")
            continue

        # Use actual lat/lon from first successfully loaded file
        global LAT, LON
        if LAT is None:
            LAT, LON = lat_file, lon_file
            print(f"  [COORDS] lat {LAT[0]:.2f}…{LAT[-1]:.2f} ({len(LAT)})"
                  f"  lon {LON[0]:.2f}…{LON[-1]:.2f} ({len(LON)})")

        print(f"  Conditioning: {cond_years[0]}–{cond_years[-1]}"
              f"  shape={tuple(cond_tensor.shape)}")

        # -- generation: ensemble of args.members members --------------------
        print(f"  Generating ensemble of {args.members} members "
              f"({len(cond_years)} years each, "
              f"batch={args.batch_size}, steps={args.sample_steps}) …")
        members = []
        for m in range(args.members):
            print(f"    member {m + 1}/{args.members} …")
            gen_norm = generate_timeseries(
                model, scheduler, cond_tensor,
                device, dtype, args.sample_steps, args.batch_size, seed=m,
                guidance_co2=args.guidance_co2, guidance_sul=args.guidance_sul,
                force_cfg=args.force_cfg,
                autocast_dtype=autocast_dt,
            )
            members.append(gen_norm * 21.0 + 4.5)   # denormalise → °C

        gen_ensemble = np.stack(members, axis=0)     # (N_ENS, T, H, W)
        gen_celsius  = gen_ensemble.mean(axis=0)     # (T, H, W) ensemble mean

        # -- model baseline (ensemble mean 1850-1900) ------------------------
        mask_bl = (cond_years >= BASELINE_START) & (cond_years <= BASELINE_END)
        gen_baseline_map = gen_celsius[mask_bl].mean(axis=0) if mask_bl.any() else None
        if gen_baseline_map is not None:
            print(f"  [MODEL BASELINE]  mean={gen_baseline_map.mean():.2f}°C"
                  f"  std={gen_baseline_map.std():.2f}°C")

        # -- if CESM2 baseline not yet set, fall back to model hist ---------
        if baseline_map is None and name == "hist":
            baseline_map = gen_baseline_map
            print(f"  [BASELINE set from model hist]  mean={baseline_map.mean():.2f}°C")

        # -- CESM2 actual data (ensemble) ----------------------------------
        cesm_years_exp, cesm_data_exp, cesm_ens_exp = None, None, None
        try:
            cesm_years_exp, cesm_ens_exp = load_cesm2_ensemble(
                exp["data_dir"], exp["realizations"], exp["time_dim"],
                exp.get("target_var", TARGET_VAR),
            )
            cesm_data_exp = cesm_ens_exp.mean(axis=0)   # (T, H, W) ensemble mean
            print(f"  CESM2: {cesm_years_exp[0]}–{cesm_years_exp[-1]}"
                  f"  ({cesm_ens_exp.shape[0]} members)")
        except Exception as e:
            import traceback
            print(f"  [WARN] CESM2 data NOT loaded for {name!r} "
                  f"(data_dir={exp['data_dir']}): {type(e).__name__}: {e}")
            traceback.print_exc()
            # If user asked for this experiment explicitly, fail loudly instead
            # of silently producing a TREFHT_<name>.nc with no CESM2 reference.
            if args.experiments and name in args.experiments:
                raise RuntimeError(
                    f"CESM2 load failed for explicit experiment {name!r}; "
                    f"refusing to write NetCDF without reference data"
                ) from e

        # -- save NetCDF ---------------------------------------------------
        if baseline_map is not None:
            nc_out = os.path.join(args.output_dir, f"TREFHT_{name}.nc")
            print(f"  Saving NetCDF …")
            save_netcdf(
                name             = name,
                gen_ensemble     = gen_ensemble,
                gen_years        = cond_years,
                baseline_map     = baseline_map,
                cesm_ensemble    = cesm_ens_exp,
                cesm_years       = cesm_years_exp,
                out_path         = nc_out,
                ckpt_path        = ckpt_path,
                gen_baseline_map = gen_baseline_map,
            )

        # -- anomaly maps --------------------------------------------------
        if baseline_map is not None:
            map_out = os.path.join(args.output_dir, f"anomaly_maps_{name}.png")
            print(f"  Plotting anomaly maps …")
            plot_anomaly_maps(
                name         = name,
                gen_data     = gen_celsius,
                gen_years    = cond_years,
                baseline_map = baseline_map,
                map_years    = exp["map_years"],
                cesm_data    = cesm_data_exp,
                cesm_years   = cesm_years_exp,
                out_path     = map_out,
                gen_ensemble = gen_ensemble,
                cesm_ensemble= cesm_ens_exp,
            )

        # -- global-mean anomaly per ensemble member -------------------------
        if baseline_map is not None:
            bl_scalar = float(area_weighted_gmean(baseline_map[np.newaxis], LAT)[0])
        else:
            gen_gmean_tmp = area_weighted_gmean(gen_celsius, LAT)
            bl_scalar = float(gen_gmean_tmp[(cond_years >= BASELINE_START) &
                                            (cond_years <= BASELINE_END)].mean())

        # (N_ENS, T) — one global-mean anomaly time series per member
        gen_anom_ens = np.stack(
            [area_weighted_gmean(gen_ensemble[m], LAT) - bl_scalar
             for m in range(gen_ensemble.shape[0])],
            axis=0,
        )

        cesm_anom_ens = cesm_anom = cesm_years_out = None
        if cesm_ens_exp is not None:
            # (N_CESM, T) global-mean anomaly per CESM2 member
            cesm_anom_ens = np.stack(
                [area_weighted_gmean(cesm_ens_exp[m], LAT) - bl_scalar
                 for m in range(cesm_ens_exp.shape[0])],
                axis=0,
            )
            cesm_anom     = cesm_anom_ens.mean(axis=0)   # ensemble mean
            cesm_years_out = cesm_years_exp

        # -- spatial IG attribution maps (per output location) ----------------
        if not args.skip_ig and name in IG_WINDOWS:
            print(f"  Computing per-location IG maps "
                  f"(n_ig_steps={args.ig_n_steps}, batch={args.ig_batch_size}) …")
            # Disable model parameter gradients — only cond gradients needed
            for p in model.parameters():
                p.requires_grad_(False)
            ig_maps = compute_ig_per_output_location(
                model            = model,
                scheduler        = scheduler,
                cond_tensor      = cond_tensor,
                lat              = LAT,
                lon              = LON,
                device           = device,
                dtype            = dtype,
                years            = cond_years,
                windows          = IG_WINDOWS[name],
                output_locations = OUTPUT_LOCATIONS,
                n_ig_steps       = args.ig_n_steps,
                batch_size       = args.ig_batch_size,
            )
            # Re-enable gradients for any subsequent training calls
            for p in model.parameters():
                p.requires_grad_(True)
            ig_prefix = os.path.join(args.output_dir, f"ig_loc_{name}")
            plot_ig_per_location(name, ig_maps, ig_prefix)

        # -- saliency maps (per output location) ------------------------------
        if not args.skip_saliency and name in IG_WINDOWS:
            print(f"  Computing per-location saliency maps "
                  f"(batch={args.saliency_batch_size}) …")
            for p in model.parameters():
                p.requires_grad_(False)
            sal_maps = compute_saliency_per_output_location(
                model            = model,
                scheduler        = scheduler,
                cond_tensor      = cond_tensor,
                lat              = LAT,
                lon              = LON,
                device           = device,
                dtype            = dtype,
                years            = cond_years,
                windows          = IG_WINDOWS[name],
                output_locations = OUTPUT_LOCATIONS,
                batch_size       = args.saliency_batch_size,
            )
            for p in model.parameters():
                p.requires_grad_(True)
            sal_prefix = os.path.join(args.output_dir, f"saliency_loc_{name}")
            plot_saliency_per_location(name, sal_maps, sal_prefix)

        # -- raw CO2 for TCRE plot (hist + projection scenarios) -------------
        co2_years_raw = co2_annual_raw = None
        if name in ("hist", "ssp370", "ssp126") and LAT is not None:
            co2_years_raw, co2_annual_raw = load_co2_global_annual(
                exp["cond_file"], exp["time_dim"], LAT
            )

        # -- internal-variability reference (hist / ssp370 only) ------------
        # Load two training members and compute their ΔT difference so that
        # the bias panels can show a ±|member-diff| grey band for context.
        ref_diff = ref_years_out = None
        if name in REF_REALIZATIONS and LAT is not None:
            try:
                ref_y, ref_ens = load_cesm2_ensemble(
                    exp["data_dir"], REF_REALIZATIONS[name], exp["time_dim"],
                    exp.get("target_var", TARGET_VAR),
                )
                ref_anom_ens = np.stack(
                    [area_weighted_gmean(ref_ens[m], LAT) - bl_scalar
                     for m in range(ref_ens.shape[0])],
                    axis=0,
                )  # (2, T)
                ref_diff     = ref_anom_ens[0] - ref_anom_ens[1]   # (T,)
                ref_years_out = ref_y
                print(f"  [REF] internal-variability diff  rms={np.sqrt((ref_diff**2).mean()):.3f}°C")
            except Exception as exc:
                print(f"  [REF] could not load reference members: {exc}")

        timeseries_results[name] = dict(
            gen_anom_ens  = gen_anom_ens,
            gen_years     = cond_years,
            cesm_anom_ens = cesm_anom_ens,
            cesm_anom     = cesm_anom,
            cesm_years    = cesm_years_out,
            color         = exp["color"],
            co2_years     = co2_years_raw,
            co2_annual    = co2_annual_raw,
            ref_diff      = ref_diff,
            ref_years     = ref_years_out,
        )

    # ── combined time series plot ──────────────────────────────────────────
    # Skip combined plots when --experiments filter is in effect; otherwise
    # we'd overwrite an existing global_mean_anomaly.png with only the
    # filtered subset. Use replot_eval.py afterwards to regenerate combined
    # plots from all TREFHT_*.nc files in output_dir.
    #
    # When sharded across ranks, each rank only holds its own subset.  Dump
    # the partial results to a per-rank pickle, then on rank 0 wait for the
    # other ranks' pickles and merge before plotting.  Without this, each
    # rank races to write the same global_mean_anomaly.png / tcre.png with
    # only its subset — the last finisher wins and the TCRE plot fails
    # entirely on ranks that lack hist or any projection.
    if timeseries_results and not args.experiments:
        if args.n_shards > 1:
            import pickle, time
            shard_dir = os.path.join(args.output_dir, "_shards")
            os.makedirs(shard_dir, exist_ok=True)
            shard_pkl = os.path.join(shard_dir, f"rank{args.shard_rank}.pkl")
            tmp_pkl = shard_pkl + ".tmp"
            with open(tmp_pkl, "wb") as f:
                pickle.dump(timeseries_results, f)
            os.replace(tmp_pkl, shard_pkl)
            print(f"[SHARD] rank={args.shard_rank} wrote {shard_pkl}")

            if args.shard_rank != 0:
                print(f"[SHARD] rank={args.shard_rank} done; rank 0 will aggregate")
            else:
                expected = [os.path.join(shard_dir, f"rank{r}.pkl")
                            for r in range(args.n_shards)]
                deadline = time.time() + 1800  # 30 min
                while True:
                    missing = [p for p in expected if not os.path.exists(p)]
                    if not missing:
                        break
                    if time.time() > deadline:
                        print(f"[SHARD] timeout waiting for {missing} — "
                              f"aggregating partial results")
                        break
                    time.sleep(15)

                merged = {}
                for p in expected:
                    if not os.path.exists(p):
                        continue
                    with open(p, "rb") as f:
                        merged.update(pickle.load(f))
                print(f"[SHARD] rank 0 merged experiments: {list(merged)}")
                timeseries_results = merged

        if args.shard_rank == 0 or args.n_shards == 1:
            ts_out  = os.path.join(args.output_dir, "global_mean_anomaly.png")
            csv_out = os.path.join(args.output_dir, "global_mean_anomaly.csv")
            print(f"\n[PLOT] Time series → {ts_out}")
            plot_timeseries(timeseries_results, ts_out)
            print(f"[CSV]  Global anomaly + bias → {csv_out}")
            save_csv(timeseries_results, csv_out)

            tcre_out = os.path.join(args.output_dir, "tcre.png")
            print(f"[PLOT] TCRE → {tcre_out}")
            plot_tcre(timeseries_results, tcre_out)

    print("\n[DONE] All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
