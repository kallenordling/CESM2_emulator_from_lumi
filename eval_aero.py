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
PROJ_ROOT   = "/projappl/project_462001112/CESM2_emulator_from_lumi"
SCRATCH     = "/scratch/project_462001112/emulator_data"
RUNS_DIR    = os.path.join(PROJ_ROOT, "runs")
DATA_ROOT   = os.path.join(SCRATCH, "training_data/TREFHT")
EMIS_DIR    = SCRATCH
CONFIG_PATH = "configs/config_aero.yaml"

# ── experiment definitions ─────────────────────────────────────────────────────
EXPERIMENTS = [
    dict(
        name        = "hist",
        data_dir    = os.path.join(DATA_ROOT, "hist"),
        cond_file   = os.path.join(EMIS_DIR, "emissions_hist_timefixed.nc"),
        realization = "LE2-1001.001",
        time_dim    = "time",
        map_years   = [1900, 2000, 2014],   # last available year instead of 2100
        color       = "#1f77b4",
    ),
    dict(
        name        = "ssp370",
        data_dir    = os.path.join(DATA_ROOT, "ssp370"),
        cond_file   = os.path.join(EMIS_DIR, "emissions_ssp370_timefixed.nc"),
        realization = "LE2-1001.001",
        time_dim    = "time",
        map_years   = [2015, 2050, 2100],
        color       = "#d62728",
    ),
    dict(
        name        = "aaer",
        data_dir    = os.path.join(DATA_ROOT, "AAER"),
        cond_file   = os.path.join(EMIS_DIR, "emissions_aero_only_timefixed.nc"),
        realization = "001",
        time_dim    = "time",
        map_years   = [1900, 2000, 2100],
        color       = "#ff7f0e",
    ),
    dict(
        name        = "ghg",
        data_dir    = os.path.join(DATA_ROOT, "GHG"),
        cond_file   = os.path.join(EMIS_DIR, "emissions_ghg_only_timefixed.nc"),
        realization = "001",
        time_dim    = "time",
        map_years   = [1900, 2000, 2100],
        color       = "#2ca02c",
    ),
]

BASELINE_START = 1850
BASELINE_END   = 1900
SAMPLE_STEPS   = 100          # fewer steps than training → faster inference
BATCH_SIZE     = 16           # years per GPU batch
COND_VARS      = ["CO2", "SUL"]
TARGET_VAR     = "TREFHT"
LAT  = np.linspace(-90, 90, 192)
LON  = np.linspace(0, 360, 288)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(runs_dir: str) -> str:
    """Return path to the highest-epoch checkpoint in runs_dir."""
    pattern = os.path.join(runs_dir, "run_multi_experiment2_*.pt")
    paths = glob.glob(pattern)
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
    """
    raw = xr.open_dataset(cond_file, chunks={time_dim: -1})[cond_vars]
    norm = raw.map(normalize)

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
    return cond_tensor, years


def generate_timeseries(
    model: UNetModel3D,
    scheduler: ContinuousDDPM,
    cond_tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    sample_steps: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Diffusion sampling for every year in cond_tensor.

    Args:
        cond_tensor: (n_vars, T, H, W) normalised conditioning on CPU
    Returns:
        numpy array (T, H, W) in *normalised* model output space
    """
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


def load_cesm2_annual(data_dir: str, realization: str, time_dim: str) -> tuple:
    """Load CESM2 TREFHT for one realization, return (years, data_celsius array).

    data_celsius shape: (T, 192, 288)
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


def area_weighted_gmean(data: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Area-weighted global mean.  data: (..., H, W), lat: (H,)."""
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]           # (H, 1)
    w /= w.mean()
    return (data * w).mean(axis=(-2, -1))                # (...,)


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
    gen_celsius: np.ndarray,
    gen_years: np.ndarray,
    baseline_map: np.ndarray,
    cesm_data: np.ndarray | None,
    cesm_years: np.ndarray | None,
    out_path: str,
    ckpt_path: str,
):
    """Save model output (and optionally CESM2 reference) to a NetCDF file.

    Variables written
    -----------------
    TREFHT_model       (year, lat, lon)  — absolute temperature [°C]
    TREFHT_model_anom  (year, lat, lon)  — anomaly re 1850-1900 baseline [°C]
    TREFHT_model_gmean (year,)           — area-weighted global mean [°C]
    TREFHT_model_gmean_anom (year,)      — global-mean anomaly [°C]
    baseline_map       (lat, lon)        — 1850-1900 climatology used [°C]

    If cesm_data is provided, also writes:
    TREFHT_cesm        (cesm_year, lat, lon)
    TREFHT_cesm_anom   (cesm_year, lat, lon)
    TREFHT_cesm_gmean  (cesm_year,)
    TREFHT_cesm_gmean_anom (cesm_year,)
    """
    coords_model = {"year": gen_years, "lat": LAT, "lon": LON}

    w = np.cos(np.deg2rad(LAT))[:, np.newaxis]
    w = w / w.mean()
    gmean_model      = (gen_celsius * w).mean(axis=(-2, -1))
    bl_scalar        = float((baseline_map * w).mean())
    anom_model       = gen_celsius - baseline_map          # (T, H, W)
    gmean_model_anom = gmean_model - bl_scalar

    ds = xr.Dataset(
        {
            "TREFHT_model": xr.DataArray(
                gen_celsius, dims=["year", "lat", "lon"], coords=coords_model,
                attrs={"units": "degC", "long_name": "Model TREFHT"}),
            "TREFHT_model_anom": xr.DataArray(
                anom_model, dims=["year", "lat", "lon"], coords=coords_model,
                attrs={"units": "degC", "long_name": "Model TREFHT anomaly re 1850-1900"}),
            "TREFHT_model_gmean": xr.DataArray(
                gmean_model, dims=["year"], coords={"year": gen_years},
                attrs={"units": "degC", "long_name": "Model global-mean TREFHT"}),
            "TREFHT_model_gmean_anom": xr.DataArray(
                gmean_model_anom, dims=["year"], coords={"year": gen_years},
                attrs={"units": "degC", "long_name": "Model global-mean TREFHT anomaly re 1850-1900"}),
            "baseline_map": xr.DataArray(
                baseline_map, dims=["lat", "lon"], coords={"lat": LAT, "lon": LON},
                attrs={"units": "degC", "long_name": "1850-1900 climatological mean"}),
        },
        attrs={
            "experiment":  name,
            "checkpoint":  os.path.basename(ckpt_path),
            "baseline":    f"{BASELINE_START}-{BASELINE_END}",
            "description": "CESM2 aerosol emulator evaluation output",
        },
    )

    if cesm_data is not None and cesm_years is not None:
        coords_cesm = {"cesm_year": cesm_years, "lat": LAT, "lon": LON}
        gmean_cesm      = (cesm_data * w).mean(axis=(-2, -1))
        anom_cesm       = cesm_data - baseline_map
        gmean_cesm_anom = gmean_cesm - bl_scalar
        ds["TREFHT_cesm"] = xr.DataArray(
            cesm_data, dims=["cesm_year", "lat", "lon"], coords=coords_cesm,
            attrs={"units": "degC", "long_name": "CESM2 TREFHT (single member)"})
        ds["TREFHT_cesm_anom"] = xr.DataArray(
            anom_cesm, dims=["cesm_year", "lat", "lon"], coords=coords_cesm,
            attrs={"units": "degC", "long_name": "CESM2 TREFHT anomaly re 1850-1900"})
        ds["TREFHT_cesm_gmean"] = xr.DataArray(
            gmean_cesm, dims=["cesm_year"], coords={"cesm_year": cesm_years},
            attrs={"units": "degC", "long_name": "CESM2 global-mean TREFHT"})
        ds["TREFHT_cesm_gmean_anom"] = xr.DataArray(
            gmean_cesm_anom, dims=["cesm_year"], coords={"cesm_year": cesm_years},
            attrs={"units": "degC", "long_name": "CESM2 global-mean TREFHT anomaly re 1850-1900"})

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
        ax_top.plot(d["gen_years"], d["gen_anom"],
                    color=c, lw=1.8, label=f"{name} (model)")
        if d.get("cesm_anom") is not None:
            ax_top.plot(d["cesm_years"], d["cesm_anom"],
                        color=c, lw=1.0, ls="--", alpha=0.6, label=f"{name} (CESM2 member)")

            # difference on common years
            common, idx_gen, idx_cs = np.intersect1d(
                d["gen_years"], d["cesm_years"], return_indices=True
            )
            diff = d["gen_anom"][idx_gen] - d["cesm_anom"][idx_cs]
            ax_bot.plot(common, diff, color=c, lw=1.5, label=name)
            ax_bot.fill_between(common, diff, alpha=0.12, color=c)

    ax_top.axhline(0, color="k", lw=0.6, ls=":")
    ax_top.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12, label="baseline period")
    ax_top.set_ylabel("TREFHT anomaly (°C)")
    ax_top.set_title("Global-mean temperature anomaly vs 1850–1900\n(model solid, CESM2 single member dashed)")
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
                      out_path: str):
    """Spatial anomaly maps at requested years.

    gen_data    : (T, H, W) generated temperature [°C]
    baseline_map: (H, W)    time-mean over 1850-1900 from hist

    Rows:
      0 — Model anomaly  (re 1850-1900)
      1 — CESM2 anomaly  (re 1850-1900)   [only if cesm_data provided]
      2 — Difference: Model − CESM2        [only if cesm_data provided]
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
        if USE_CARTOPY:
            im = ax.pcolormesh(LON, LAT, data, cmap=cmap, norm=norm,
                               transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, lw=0.5)
        else:
            im = ax.imshow(data, origin="lower", aspect="auto",
                           cmap=cmap, norm=norm,
                           extent=[0, 360, -90, 90])
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.75, label="°C")

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

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f"TREFHT anomaly vs 1850–1900 — {name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",   default=RUNS_DIR)
    parser.add_argument("--output-dir", default="eval_output")
    parser.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS)
    parser.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32  # float32 avoids dtype conflicts in SinusoidalPosEmb
    print(f"[DEVICE] {device}  dtype={dtype}")

    # ── load model ─────────────────────────────────────────────────────────
    ckpt_path = find_latest_checkpoint(args.runs_dir)
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
    print("\n[BASELINE] Loading hist CESM2 data to compute 1850–1900 mean …")
    hist_exp = next(e for e in EXPERIMENTS if e["name"] == "hist")
    try:
        cesm_hist_years, cesm_hist_data = load_cesm2_annual(
            hist_exp["data_dir"], hist_exp["realization"], hist_exp["time_dim"]
        )
        mask_bl = (cesm_hist_years >= BASELINE_START) & (cesm_hist_years <= BASELINE_END)
        baseline_map = cesm_hist_data[mask_bl].mean(axis=0)       # (H, W)  in °C
        print(f"  baseline map  mean={baseline_map.mean():.2f}°C  std={baseline_map.std():.2f}°C")
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
            cond_tensor, cond_years = build_cond_tensor(
                exp["cond_file"], COND_VARS, exp["time_dim"],
                pca_cond, N_COMP_COND,
            )
        except Exception as e:
            print(f"  SKIP (conditioning failed): {e}")
            continue

        print(f"  Conditioning: {cond_years[0]}–{cond_years[-1]}"
              f"  shape={tuple(cond_tensor.shape)}")

        # -- generation ----------------------------------------------------
        print(f"  Generating {len(cond_years)} years "
              f"(batch={args.batch_size}, steps={args.sample_steps}) …")
        gen_norm = generate_timeseries(
            model, scheduler, cond_tensor,
            device, dtype, args.sample_steps, args.batch_size,
        )
        # denormalise: DENORM_FN["TREFHT"] = lambda x: x * 21.0 + 4.5
        gen_celsius = gen_norm * 21.0 + 4.5          # (T, H, W)

        # -- if baseline_map not yet set, use first-pass of hist ------------
        if baseline_map is None and name == "hist":
            mask_bl = (cond_years >= BASELINE_START) & (cond_years <= BASELINE_END)
            baseline_map = gen_celsius[mask_bl].mean(axis=0)
            print(f"  [BASELINE set from model hist]  mean={baseline_map.mean():.2f}°C")

        # -- CESM2 actual data ---------------------------------------------
        cesm_years_exp, cesm_data_exp = None, None
        try:
            cesm_years_exp, cesm_data_exp = load_cesm2_annual(
                exp["data_dir"], exp["realization"], exp["time_dim"]
            )
            print(f"  CESM2: {cesm_years_exp[0]}–{cesm_years_exp[-1]}")
        except Exception as e:
            print(f"  CESM2 data not loaded: {e}")

        # -- save NetCDF ---------------------------------------------------
        if baseline_map is not None:
            nc_out = os.path.join(args.output_dir, f"TREFHT_{name}.nc")
            print(f"  Saving NetCDF …")
            save_netcdf(
                name         = name,
                gen_celsius  = gen_celsius,
                gen_years    = cond_years,
                baseline_map = baseline_map,
                cesm_data    = cesm_data_exp,
                cesm_years   = cesm_years_exp,
                out_path     = nc_out,
                ckpt_path    = ckpt_path,
            )

        # -- anomaly maps --------------------------------------------------
        if baseline_map is not None:
            map_out = os.path.join(args.output_dir, f"anomaly_maps_{name}.png")
            print(f"  Plotting anomaly maps …")
            plot_anomaly_maps(
                name        = name,
                gen_data    = gen_celsius,
                gen_years   = cond_years,
                baseline_map= baseline_map,
                map_years   = exp["map_years"],
                cesm_data   = cesm_data_exp,
                cesm_years  = cesm_years_exp,
                out_path    = map_out,
            )

        # -- global-mean anomaly -------------------------------------------
        gen_gmean = area_weighted_gmean(gen_celsius, LAT)

        # baseline from hist model run (or CESM2)
        if baseline_map is not None:
            bl_scalar = float(area_weighted_gmean(baseline_map[np.newaxis], LAT)[0])
        else:
            bl_scalar = float(gen_gmean[(cond_years >= BASELINE_START) &
                                        (cond_years <= BASELINE_END)].mean())

        gen_anom = gen_gmean - bl_scalar

        cesm_anom = cesm_years_out = None
        if cesm_data_exp is not None:
            cesm_gmean = area_weighted_gmean(cesm_data_exp, LAT)
            cesm_anom  = cesm_gmean - bl_scalar
            cesm_years_out = cesm_years_exp

        timeseries_results[name] = dict(
            gen_anom   = gen_anom,
            gen_years  = cond_years,
            cesm_anom  = cesm_anom,
            cesm_years = cesm_years_out,
            color      = exp["color"],
        )

    # ── combined time series plot ──────────────────────────────────────────
    if timeseries_results:
        ts_out = os.path.join(args.output_dir, "global_mean_anomaly.png")
        print(f"\n[PLOT] Time series → {ts_out}")
        plot_timeseries(timeseries_results, ts_out)

    print("\n[DONE] All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
