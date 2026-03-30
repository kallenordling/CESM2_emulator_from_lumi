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
        realizations = ["LE2-1001.001", "LE2-1011.001", "LE2-1021.002",
                        "LE2-1031.002", "LE2-1041.003"],
        time_dim     = "time",
        map_years    = [1900, 2000, 2014],   # last available year instead of 2100
        color        = "#1f77b4",
    ),
    dict(
        name         = "ssp370",
        data_dir     = os.path.join(DATA_ROOT, "ssp370"),
        cond_file    = os.path.join(EMIS_DIR, "emissions_ssp370_timefixed.nc"),
        realizations = ["LE2-1001.001", "LE2-1011.001", "LE2-1021.002",
                        "LE2-1031.002", "LE2-1041.003"],
        time_dim     = "time",
        map_years    = [2015, 2050, 2100],
        color        = "#d62728",
    ),
    dict(
        name         = "aaer",
        data_dir     = os.path.join(DATA_ROOT, "AAER"),
        cond_file    = os.path.join(EMIS_DIR, "emissions_aero_only_timefixed.nc"),
        realizations = ["001", "002", "003", "004", "005"],
        time_dim     = "time",
        map_years    = [1900, 2000, 2100],
        color        = "#ff7f0e",
    ),
    dict(
        name         = "ghg",
        data_dir     = os.path.join(DATA_ROOT, "GHG"),
        cond_file    = os.path.join(EMIS_DIR, "emissions_ghg_only_timefixed.nc"),
        realizations = ["001", "002", "003", "004", "005"],
        time_dim     = "time",
        map_years    = [1900, 2000, 2100],
        color        = "#2ca02c",
    ),
]

BASELINE_START = 1850
BASELINE_END   = 1900
SAMPLE_STEPS   = 100          # fewer steps than training → faster inference
BATCH_SIZE     = 16           # years per GPU batch
N_ENSEMBLE     = 5            # diffusion samples per experiment
COND_VARS      = ["CO2", "SUL"]
TARGET_VAR     = "TREFHT"
LAT  = None   # set from first conditioning file in main()
LON  = None   # set from first conditioning file in main()

NULL_COND = -1.0   # CFG null value (pre-industrial baseline under normalisation)

# Time windows for spatial IG maps, keyed by experiment name
IG_WINDOWS = {
    "hist":   [(1920, 1960, "1920–1960"), (1960, 1990, "1960–1990"), (1990, 2014, "1990–2014")],
    "ssp370": [(2020, 2050, "2020–2050"), (2050, 2080, "2050–2080"), (2080, 2100, "2080–2100")],
    "aaer":   [(1920, 1960, "1920–1960"), (1960, 1990, "1960–1990"), (1990, 2100, "1990–2100")],
    "ghg":    [(1920, 1970, "1920–1970"), (1970, 2020, "1970–2020"), (2050, 2100, "2050–2100")],
}


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
                      pca_objects, n_components_cond):
    """Load, normalize, optionally PCA-project the conditioning data.

    Returns:
        cond_tensor : torch.Tensor  (n_vars, T, H, W)
        years       : np.ndarray    (T,) integer years
        lat         : np.ndarray    (H,) latitude values from file
        lon         : np.ndarray    (W,) longitude values from file
    """
    raw = xr.open_dataset(cond_file, chunks={time_dim: -1})[cond_vars]
    norm = raw.map(normalize)

    lat = norm["lat"].values.astype(np.float64)
    lon = norm["lon"].values.astype(np.float64)

    # to_stacked_array needs ("var", time_dim, "lat", "lon")
    stacked = norm.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"])
    stacked = stacked.transpose("var", time_dim, "lat", "lon")
    cond_tensor = torch.tensor(stacked.values, dtype=torch.float32)

    years = extract_years(norm[time_dim].values)

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
) -> np.ndarray:
    """Diffusion sampling for every year in cond_tensor.

    Args:
        cond_tensor: (n_vars, T, H, W) normalised conditioning on CPU
        seed: optional RNG seed for reproducible ensemble members
    Returns:
        numpy array (T, H, W) in *normalised* model output space
    """
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

        with torch.no_grad():
            for step_idx, t_idx in enumerate(scheduler.timesteps):
                t = scheduler.log_snr(steps[t_idx]).expand(B).to(dtype)
                pred = model(gen, t, cond_map=cond_b)
                gen  = scheduler.step(pred, timestep=t_idx, sample=gen).prev_sample

        results.append(gen.squeeze(1).squeeze(1).cpu().float())   # (B, H, W)

    return torch.cat(results, dim=0).numpy()   # (T, H, W)


def load_cesm2_annual_single(data_dir: str, realization: str, time_dim: str) -> tuple:
    """Load CESM2 TREFHT for one realization, return (years, data_celsius array).

    data_celsius shape: (T, lat, lon)
    """
    path = os.path.join(data_dir, realization, "*.nc")
    ds = xr.open_mfdataset(path, combine="by_coords",
                           chunks={time_dim: 50})[TARGET_VAR]

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


def load_cesm2_ensemble(data_dir: str, realizations: list, time_dim: str) -> tuple:
    """Load CESM2 TREFHT for multiple realizations.

    Returns:
        years           : np.ndarray (T,) — years from first successfully loaded member
        cesm_ensemble   : np.ndarray (N, T, lat, lon) — all members on common years
    """
    members = []
    common_years = None
    for real in realizations:
        try:
            yrs, data = load_cesm2_annual_single(data_dir, real, time_dim)
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


def compute_ig_spatial_maps(
    model: UNetModel3D,
    scheduler: ContinuousDDPM,
    cond_tensor: torch.Tensor,
    lat: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    years: np.ndarray,
    windows: list,
    n_ig_steps: int = 30,
    batch_size: int = 8,
    t_proxy: float = 0.2,
    seed: int = 42,
) -> dict:
    """Compute spatially-resolved Integrated Gradients for CO2 and SUL conditioning.

    Differentiates the area-weighted global-mean temperature prediction w.r.t.
    the conditioning map at each grid point, keeping the full spatial structure.

    The resulting maps answer: "which grid points of the conditioning inputs
    does the model pay most attention to?"

    Parameters
    ----------
    windows : list of (year_start, year_end, label) tuples
        IG maps are averaged over each window and returned separately.

    Returns
    -------
    dict keyed by window label → {"co2": (H, W), "sul": (H, W)} absolute IG maps
    """
    _, T_total, H, W = cond_tensor.shape

    w_lat = torch.tensor(np.cos(np.deg2rad(lat)), dtype=dtype, device=device)
    w_lat = (w_lat / w_lat.mean()).view(H, 1)

    # Accumulate full (T, H, W) spatial IG maps
    ig_co2_full = np.zeros((T_total, H, W), dtype=np.float32)
    ig_sul_full = np.zeros((T_total, H, W), dtype=np.float32)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for t_start in tqdm(range(0, T_total, batch_size), desc="  IG spatial batches"):
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

        noise = torch.randn(B, 1, 1, H, W, device=device, dtype=dtype,
                            generator=rng)
        t_val   = torch.full((B,), t_proxy, device=device, dtype=dtype)
        log_snr = scheduler.log_snr(t_val)
        x_clean = torch.zeros(B, 1, 1, H, W, device=device, dtype=dtype)
        x_noisy = scheduler.add_noise(x_clean, noise, log_snr).detach()
        log_snr = log_snr.detach()

        sum_grad_co2 = torch.zeros(B, H, W, device=device, dtype=dtype)
        sum_grad_sul = torch.zeros(B, H, W, device=device, dtype=dtype)

        for k in range(1, n_ig_steps + 1):
            alpha = k / n_ig_steps
            cond_k = (cond_null_b + alpha * delta).detach().requires_grad_(True)

            v_pred  = model(x_noisy, log_snr, cond_map=cond_k)
            pred_x0 = scheduler.predict_start_from_v(x_noisy, log_snr, v_pred)

            pred_map = pred_x0.squeeze(1).squeeze(1)          # (B, H, W)
            T_global = (pred_map * w_lat).mean(dim=(-2, -1))  # (B,)

            grads = torch.autograd.grad(T_global.sum(), cond_k)[0]  # (B, 2, 1, H, W)
            sum_grad_co2 += grads[:, 0, 0]   # (B, H, W)
            sum_grad_sul += grads[:, 1, 0]

        # IG = delta × mean_gradient  — keep spatial structure (B, H, W)
        ig_co2_full[t_start:t_end] = (
            delta[:, 0, 0].detach() * (sum_grad_co2 / n_ig_steps)
        ).cpu().numpy()
        ig_sul_full[t_start:t_end] = (
            delta[:, 1, 0].detach() * (sum_grad_sul / n_ig_steps)
        ).cpu().numpy()

    # Average |IG| over each time window
    result = {}
    for y_start, y_end, label in windows:
        mask = (years >= y_start) & (years <= y_end)
        if not mask.any():
            continue
        result[label] = {
            "co2": np.abs(ig_co2_full[mask]).mean(axis=0),   # (H, W)
            "sul": np.abs(ig_sul_full[mask]).mean(axis=0),
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

        # individual ensemble members — thin, semi-transparent
        for m in range(gen_anom_ens.shape[0]):
            da_m = xr.DataArray(
                gen_anom_ens[m], dims=["year"],
                coords={"year": d["gen_years"]},
            )
            da_m.plot.line(ax=ax_top, color=c, lw=0.7, alpha=0.35)

        # ensemble mean — thick solid line with legend entry
        da_mean = xr.DataArray(
            gen_anom_mean, dims=["year"],
            coords={"year": d["gen_years"]},
            attrs={"long_name": "TREFHT anomaly", "units": "°C"},
        )
        da_mean.plot.line(ax=ax_top, color=c, lw=2.0, label=f"{name} (model mean)")

        if d.get("cesm_anom") is not None:
            cesm_anom_ens = d["cesm_anom_ens"]           # (N_CESM, T)
            cesm_anom_mean = d["cesm_anom"]               # (T,)

            # individual CESM2 members — thin dashed, semi-transparent
            for m in range(cesm_anom_ens.shape[0]):
                da_cm = xr.DataArray(
                    cesm_anom_ens[m], dims=["year"],
                    coords={"year": d["cesm_years"]},
                )
                da_cm.plot.line(ax=ax_top, color=c, lw=0.7, ls="--", alpha=0.35)

            # CESM2 ensemble mean — thick dashed with legend entry
            da_cesm = xr.DataArray(
                cesm_anom_mean, dims=["year"],
                coords={"year": d["cesm_years"]},
                attrs={"long_name": "TREFHT anomaly", "units": "°C"},
            )
            da_cesm.plot.line(ax=ax_top, color=c, lw=2.0, ls="--", alpha=0.8,
                              label=f"{name} (CESM2 ens. mean)")

            common, idx_gen, idx_cs = np.intersect1d(
                d["gen_years"], d["cesm_years"], return_indices=True
            )
            # bias: model ensemble mean vs CESM2 ensemble mean
            diff_mean = gen_anom_mean[idx_gen] - cesm_anom_mean[idx_cs]
            da_diff = xr.DataArray(
                diff_mean, dims=["year"], coords={"year": common},
                attrs={"long_name": "Model − CESM2", "units": "°C"},
            )
            da_diff.plot.line(ax=ax_bot, color=c, lw=1.5, label=name)
            # shade model spread around bias
            diff_min = gen_anom_ens[:, idx_gen].min(axis=0) - cesm_anom_mean[idx_cs]
            diff_max = gen_anom_ens[:, idx_gen].max(axis=0) - cesm_anom_mean[idx_cs]
            ax_bot.fill_between(common, diff_min, diff_max, alpha=0.12, color=c)

    ax_top.axhline(0, color="k", lw=0.6, ls=":")
    ax_top.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12, label="baseline period")
    ax_top.set_xlabel("")
    ax_top.set_ylabel("TREFHT anomaly (°C)")
    ax_top.set_title("Global-mean temperature anomaly vs 1850–1900\n(model solid, CESM2 dashed — both 5-member ensembles)")
    ax_top.legend(fontsize=8, ncol=2)
    ax_top.grid(True, alpha=0.25)

    ax_bot.axhline(0, color="k", lw=0.9)
    ax_bot.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12)
    ax_bot.set_xlabel("Year")
    ax_bot.set_ylabel("Bias: model − CESM2 (°C)")
    ax_bot.set_title("Model bias relative to CESM2 single member")
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def _nearest_year(years: np.ndarray, target: int) -> int:
    """Return the year in `years` closest to `target`."""
    idx = np.argmin(np.abs(years - target))
    return int(years[idx])


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
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
    except ImportError:
        USE_CARTOPY = False

    has_cesm = (cesm_data is not None) and (cesm_years is not None)
    n_cols = len(map_years)
    n_rows = 3 if has_cesm else 1
    row_labels = ["Model"]
    if has_cesm:
        row_labels += ["CESM2", "Model − CESM2"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()} if USE_CARTOPY else {},
        squeeze=False,
    )

    vmax_anom = 4.0
    vmax_diff = 2.0
    cmap = plt.cm.RdBu_r
    norm_anom = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax_anom, vmax=vmax_anom)
    norm_diff = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax_diff, vmax=vmax_diff)

    def _plot_panel(ax, data, norm, title):
        # Add a cyclic longitude column so pcolormesh closes the wrap at 0°/360°
        if USE_CARTOPY:
            from cartopy.util import add_cyclic_point
            data_cyc, lon_cyc = add_cyclic_point(data, coord=LON)
        else:
            data_cyc = np.concatenate([data, data[:, :1]], axis=1)
            lon_cyc  = np.append(LON, LON[0] + 360.0)

        da = xr.DataArray(
            data_cyc, dims=["lat", "lon"],
            coords={"lat": LAT, "lon": lon_cyc},
            attrs={"units": "°C"},
        )
        plot_kwargs = dict(
            ax=ax, cmap=cmap, norm=norm,
            add_colorbar=True,
            cbar_kwargs={"label": "°C", "shrink": 0.75},
        )
        if USE_CARTOPY:
            plot_kwargs["transform"] = ccrs.PlateCarree()
        da.plot.pcolormesh(**plot_kwargs)
        if USE_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, lw=0.5)
            ax.add_feature(cfeature.BORDERS, lw=0.3, linestyle=":")
            gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                              color="grey", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
        ax.set_title(title, fontsize=9)
        # Global mean annotation in bottom-right corner
        gmean = float(area_weighted_gmean(data[np.newaxis], LAT)[0])
        ax.text(0.98, 0.03, f"GM: {gmean:+.2f}°C",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    for col, yr_target in enumerate(map_years):
        yr_gen  = _nearest_year(gen_years, yr_target)
        idx_gen = int(np.where(gen_years == yr_gen)[0][0])
        anom_gen = gen_data[idx_gen] - baseline_map           # (H, W)

        _plot_panel(axes[0, col], anom_gen, norm_anom,
                    f"{name} model  ({yr_gen})")

        if has_cesm:
            yr_cs  = _nearest_year(cesm_years, yr_target)
            idx_cs = int(np.where(cesm_years == yr_cs)[0][0])
            anom_cs = cesm_data[idx_cs] - baseline_map

            _plot_panel(axes[1, col], anom_cs, norm_anom,
                        f"{name} CESM2  ({yr_cs})")
            _plot_panel(axes[2, col], anom_gen - anom_cs, norm_diff,
                        f"Model − CESM2  ({yr_gen})")

            # Stipple where difference is significant vs natural variability
            if gen_ensemble is not None and cesm_ensemble is not None:
                # Per-member anomalies at this year: (N, H, W) and (M, H, W)
                gen_members  = gen_ensemble[:, idx_gen]  - baseline_map   # (N, H, W)
                cesm_members = cesm_ensemble[:, idx_cs] - baseline_map    # (M, H, W)
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
                if USE_CARTOPY:
                    ax_diff.scatter(sig_lons, sig_lats, transform=ccrs.PlateCarree(),
                                    zorder=5, **plot_kw)
                else:
                    ax_diff.scatter(sig_lons, sig_lats, zorder=5, **plot_kw)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f"TREFHT anomaly vs 1850–1900 — {name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_ig_spatial_maps(name: str, ig_results: dict, out_path: str):
    """Plot spatial IG attribution maps for CO2 and SUL per time window.

    Rows: CO2, SUL
    Columns: one per time window

    Bright colours indicate grid points where the model's predicted global-mean
    temperature is most sensitive to the conditioning at that location.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
    except ImportError:
        USE_CARTOPY = False

    windows = list(ig_results.keys())
    if not windows:
        print(f"  [IG] No windows to plot for {name}, skipping.")
        return

    n_cols = len(windows)
    n_rows = 2   # row 0 = CO2, row 1 = SUL

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()} if USE_CARTOPY else {},
        squeeze=False,
    )

    # Shared colour scale per channel (98th percentile across all windows)
    vmax_co2 = max(
        np.percentile(ig_results[w]["co2"], 98) for w in windows
    )
    vmax_sul = max(
        np.percentile(ig_results[w]["sul"], 98) for w in windows
    )

    def _plot_map(ax, data, vmax, title):
        if USE_CARTOPY:
            from cartopy.util import add_cyclic_point
            data_cyc, lon_cyc = add_cyclic_point(data, coord=LON)
        else:
            data_cyc = np.concatenate([data, data[:, :1]], axis=1)
            lon_cyc  = np.append(LON, LON[0] + 360.0)

        da = xr.DataArray(
            data_cyc, dims=["lat", "lon"],
            coords={"lat": LAT, "lon": lon_cyc},
        )
        plot_kwargs = dict(
            ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={"label": "|IG attr.|", "shrink": 0.75},
        )
        if USE_CARTOPY:
            plot_kwargs["transform"] = ccrs.PlateCarree()
        da.plot.pcolormesh(**plot_kwargs)
        if USE_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, lw=0.5)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                              color="grey", alpha=0.5, linestyle="--")
            gl.top_labels   = False
            gl.right_labels = False
        ax.set_title(title, fontsize=9)

    for col, window in enumerate(windows):
        _plot_map(axes[0, col], ig_results[window]["co2"], vmax_co2,
                  f"CO2 attribution\n{window}")
        _plot_map(axes[1, col], ig_results[window]["sul"], vmax_sul,
                  f"SUL attribution\n{window}")

    axes[0, 0].set_ylabel("CO2", fontsize=10)
    axes[1, 0].set_ylabel("SUL", fontsize=10)

    fig.suptitle(
        f"Spatial IG attribution — {name}\n"
        "(bright = conditioning at this grid point most influences predicted global-mean T)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


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
    parser.add_argument("--skip-ig",       action="store_true",
                        help="Skip spatial IG attribution maps (saves time/memory)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32  # float32 avoids dtype conflicts in SinusoidalPosEmb
    print(f"[DEVICE] {device}  dtype={dtype}")

    # ── load model ─────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint if args.checkpoint else find_latest_checkpoint(args.runs_dir)
    model, pca_state = load_model(ckpt_path, CONFIG_PATH, device)
    model = model.to(dtype)   # match input dtype to avoid bf16/float32 mismatch
    print(f"[PCA] {'Found in checkpoint' if pca_state else 'None — no PCA projection'}")

    pca_cond   = pca_state.get("cond")   if pca_state else None
    pca_target = pca_state.get("target") if pca_state else None

    # n_components from the first ClimateDataset (config_data.yaml value)
    N_COMP_COND = [3, 5] if pca_cond else None

    # ── build scheduler ────────────────────────────────────────────────────
    cfg = OmegaConf.load(CONFIG_PATH)
    scheduler: ContinuousDDPM = instantiate(cfg.scheduler)

    # ── compute hist baseline map (H, W) for anomaly reference ─────────────
    print("\n[BASELINE] Loading hist CESM2 ensemble to compute 1850–1900 mean …")
    hist_exp = next(e for e in EXPERIMENTS if e["name"] == "hist")
    try:
        cesm_hist_years, cesm_hist_ens = load_cesm2_ensemble(
            hist_exp["data_dir"], hist_exp["realizations"], hist_exp["time_dim"]
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

    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"[EXP] {name}")

        # -- conditioning --------------------------------------------------
        print("  Building conditioning tensor …")
        try:
            cond_tensor, cond_years, lat_file, lon_file = build_cond_tensor(
                exp["cond_file"], COND_VARS, exp["time_dim"],
                pca_cond, N_COMP_COND,
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

        # -- generation: ensemble of N_ENSEMBLE members ----------------------
        print(f"  Generating ensemble of {N_ENSEMBLE} members "
              f"({len(cond_years)} years each, "
              f"batch={args.batch_size}, steps={args.sample_steps}) …")
        members = []
        for m in range(N_ENSEMBLE):
            print(f"    member {m + 1}/{N_ENSEMBLE} …")
            gen_norm = generate_timeseries(
                model, scheduler, cond_tensor,
                device, dtype, args.sample_steps, args.batch_size, seed=m,
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
                exp["data_dir"], exp["realizations"], exp["time_dim"]
            )
            cesm_data_exp = cesm_ens_exp.mean(axis=0)   # (T, H, W) ensemble mean
            print(f"  CESM2: {cesm_years_exp[0]}–{cesm_years_exp[-1]}"
                  f"  ({cesm_ens_exp.shape[0]} members)")
        except Exception as e:
            print(f"  CESM2 data not loaded: {e}")

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
             for m in range(N_ENSEMBLE)],
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

        # -- spatial IG attribution maps -----------------------------------
        if not args.skip_ig and name in IG_WINDOWS:
            print(f"  Computing spatial IG maps "
                  f"(n_ig_steps={args.ig_n_steps}, batch={args.ig_batch_size}) …")
            # Disable model parameter gradients — only cond gradients needed
            for p in model.parameters():
                p.requires_grad_(False)
            ig_maps = compute_ig_spatial_maps(
                model        = model,
                scheduler    = scheduler,
                cond_tensor  = cond_tensor,
                lat          = LAT,
                device       = device,
                dtype        = dtype,
                years        = cond_years,
                windows      = IG_WINDOWS[name],
                n_ig_steps   = args.ig_n_steps,
                batch_size   = args.ig_batch_size,
            )
            # Re-enable gradients for any subsequent training calls
            for p in model.parameters():
                p.requires_grad_(True)
            ig_out = os.path.join(args.output_dir, f"ig_spatial_{name}.png")
            plot_ig_spatial_maps(name, ig_maps, ig_out)

        timeseries_results[name] = dict(
            gen_anom_ens  = gen_anom_ens,
            gen_years     = cond_years,
            cesm_anom_ens = cesm_anom_ens,
            cesm_anom     = cesm_anom,
            cesm_years    = cesm_years_out,
            color         = exp["color"],
        )

    # ── combined time series plot ──────────────────────────────────────────
    if timeseries_results:
        ts_out = os.path.join(args.output_dir, "global_mean_anomaly.png")
        print(f"\n[PLOT] Time series → {ts_out}")
        plot_timeseries(timeseries_results, ts_out)

    print("\n[DONE] All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
