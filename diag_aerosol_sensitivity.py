#!/usr/bin/env python3
"""diag_aerosol_sensitivity.py
===============================
Diagnostic: does the model actually respond to aerosol (SUL) conditioning?

Runs hist conditioning THREE times:
  1. Normal  — CO2 and SUL as observed
  2. NoSUL   — CO2 as observed, SUL channel set to its minimum (-1, i.e. no aerosols)
  3. CESM2   — actual CESM2 hist ensemble mean (reference)

If the model correctly uses SUL forcing, NoSUL temperatures should be
*warmer* than Normal, and the difference should peak around 1970-1980
when historical aerosol loading was highest.

The magnitude of (NoSUL − Normal) vs (Model − CESM2) tells us how much
of the warm bias can be explained by aerosol under-response.

Run on LUMI:
    python diag_aerosol_sensitivity.py [--checkpoint /path/to/run.pt]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from data.climate_dataset import (
    ClimateDataset,
    normalize,
    DENORM_FN,
    PREPROCESS_FN,
)
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from models.video_net import UNetModel3D

# ── paths (same as eval_aero.py) ──────────────────────────────────────────────
PROJ_ROOT   = "/projappl/project_462001328/CESM2_emulator_from_lumi"
SCRATCH     = "/scratch/project_462001112/emulator_data"
RUNS_DIR    = os.path.join(PROJ_ROOT, "runs")
DATA_ROOT   = os.path.join(SCRATCH, "training_data/TREFHT")
EMIS_DIR    = SCRATCH
CONFIG_PATH = "configs/config_aero.yaml"

HIST_DATA_DIR  = os.path.join(DATA_ROOT, "hist")
HIST_COND_FILE = os.path.join(EMIS_DIR, "emissions_hist_timefixed.nc")
HIST_MEMBERS   = ["LE2-1001.001", "LE2-1011.001", "LE2-1021.002",
                  "LE2-1031.002", "LE2-1041.003"]

BASELINE_START = 1850
BASELINE_END   = 1900
SAMPLE_STEPS   = 50   # fewer steps → fast diagnostic
BATCH_SIZE     = 16
N_ENSEMBLE     = 3    # small ensemble — enough to check direction
COND_VARS      = ["CO2", "SUL"]
# Channel indices in the (C, T, H, W) conditioning tensor
CO2_IDX = 0
SUL_IDX = 1


# ── helpers ───────────────────────────────────────────────────────────────────

def find_latest_checkpoint(runs_dir: str) -> str:
    paths = [p for p in glob.glob(os.path.join(runs_dir, "*.pt"))
             if not p.endswith("_best.pt")]
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {runs_dir}")
    def _epoch(p):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    best = max(paths, key=_epoch)
    print(f"[CKPT] {best}  (epoch {_epoch(best)})")
    return best


def load_model(ckpt_path, config_path, device):
    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["EMA"], strict=False)
    model = model.to(device).eval()
    return model, ckpt.get("PCA")


def extract_years(coord_vals) -> np.ndarray:
    if hasattr(coord_vals[0], "year"):
        return np.array([int(str(v)[:4]) for v in coord_vals])
    return np.asarray(coord_vals, dtype=int)


def build_cond_tensor(cond_file, time_dim="time"):
    """Return (cond_tensor (C, T, H, W), years, lat, lon) — annual resolution."""
    raw = xr.open_dataset(cond_file, chunks={time_dim: -1})[COND_VARS]
    norm = raw.map(normalize)
    lat  = norm["lat"].values.astype(np.float64)
    lon  = norm["lon"].values.astype(np.float64)
    stacked = norm.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"])
    stacked = stacked.transpose("var", time_dim, "lat", "lon")
    cond_tensor = torch.tensor(stacked.values, dtype=torch.float32)
    years = extract_years(norm[time_dim].values)
    raw.close()
    return cond_tensor, years, lat, lon


def generate_ensemble(model, scheduler, cond_tensor, device, dtype,
                      n_ensemble=N_ENSEMBLE, sample_steps=SAMPLE_STEPS,
                      batch_size=BATCH_SIZE, label=""):
    """Generate N_ENSEMBLE members → (N, T, H, W) in °C."""
    members = []
    for m in range(n_ensemble):
        print(f"  {label} member {m+1}/{n_ensemble}")
        torch.manual_seed(m)
        _, T, H, W = cond_tensor.shape
        scheduler.set_timesteps(sample_steps)
        steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        results = []
        for i in tqdm(range(0, T, batch_size), desc="    batches", leave=False):
            chunk  = cond_tensor[:, i:i+batch_size]
            B      = chunk.shape[1]
            cond_b = chunk.permute(1, 0, 2, 3).unsqueeze(2).to(device=device, dtype=dtype)
            gen    = torch.randn(B, 1, 1, H, W, device=device, dtype=dtype)
            with torch.no_grad():
                for step_idx, t_idx in enumerate(scheduler.timesteps):
                    t    = scheduler.log_snr(steps[t_idx]).expand(B).to(dtype)
                    pred = model(gen, t, cond_map=cond_b)
                    gen  = scheduler.step(pred, timestep=t_idx, sample=gen).prev_sample
            results.append(gen.squeeze(1).squeeze(1).cpu().float())
        gen_norm = torch.cat(results, dim=0).numpy()
        members.append(gen_norm * 21.0 + 4.5)   # denorm → °C
    return np.stack(members, axis=0)             # (N, T, H, W)


def area_weighted_gmean(data: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """data: (..., H, W)  →  (...,)"""
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    w /= w.mean()
    return (data * w).mean(axis=(-2, -1))


def load_cesm2_ensemble(data_dir, realizations, time_dim="time"):
    members, common_years = [], None
    for real in realizations:
        path = os.path.join(data_dir, real, "*.nc")
        ds   = xr.open_mfdataset(path, combine="by_coords",
                                  chunks={time_dim: 50})["TREFHT"]
        ds   = ds - 273.15
        try:
            ds_ann = ds.resample(time="YE").mean().compute()
            yrs    = extract_years(ds_ann.time.values)
            data   = ds_ann.values.astype(np.float32)
        except Exception:
            ds.load()
            yrs  = extract_years(ds[time_dim].values)
            data = ds.values.astype(np.float32)

        if common_years is None:
            common_years = yrs
            members.append(data)
        else:
            common, ic, im = np.intersect1d(common_years, yrs, return_indices=True)
            members = [m[ic] for m in members]
            members.append(data[im])
            common_years = common
    return common_years, np.stack(members, axis=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",    default=RUNS_DIR)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--output-dir",  default="/scratch/project_462001328/eval_output")
    parser.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS)
    parser.add_argument("--n-ensemble",   type=int, default=N_ENSEMBLE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    print(f"[DEVICE] {device}")

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.runs_dir)
    model, _  = load_model(ckpt_path, CONFIG_PATH, device)
    model     = model.to(dtype)

    cfg       = OmegaConf.load(CONFIG_PATH)
    scheduler = instantiate(cfg.scheduler)

    # ── conditioning tensors ───────────────────────────────────────────────
    print("\n[COND] Building hist conditioning …")
    cond_normal, years, lat, lon = build_cond_tensor(HIST_COND_FILE, time_dim="time")

    # NoSUL: same CO2 channel, SUL channel set to its normalised minimum (-1)
    cond_nosul = cond_normal.clone()
    sul_min_norm = float(cond_normal[SUL_IDX].min())   # actual min in norm space
    cond_nosul[SUL_IDX] = sul_min_norm
    print(f"  Normal SUL range (normalised): "
          f"[{float(cond_normal[SUL_IDX].min()):.3f}, "
          f"{float(cond_normal[SUL_IDX].max()):.3f}]")
    print(f"  NoSUL SUL value: {sul_min_norm:.3f}  (constant, = dataset minimum)")

    # ── print CO2 channel stats to verify it's the same ───────────────────
    print(f"  CO2 range (normalised): "
          f"[{float(cond_normal[CO2_IDX].min()):.3f}, "
          f"{float(cond_normal[CO2_IDX].max()):.3f}]")

    # ── generate ──────────────────────────────────────────────────────────
    print("\n[GEN] Normal conditioning …")
    ens_normal = generate_ensemble(model, scheduler, cond_normal, device, dtype,
                                   n_ensemble=args.n_ensemble,
                                   sample_steps=args.sample_steps,
                                   label="Normal")

    print("\n[GEN] No-SUL conditioning (SUL at minimum) …")
    ens_nosul  = generate_ensemble(model, scheduler, cond_nosul, device, dtype,
                                   n_ensemble=args.n_ensemble,
                                   sample_steps=args.sample_steps,
                                   label="NoSUL")

    # ── load CESM2 reference ───────────────────────────────────────────────
    print("\n[CESM2] Loading hist ensemble …")
    try:
        cesm_years, cesm_ens = load_cesm2_ensemble(HIST_DATA_DIR, HIST_MEMBERS)
        cesm_mean = cesm_ens.mean(axis=0)  # (T, H, W)
    except Exception as e:
        print(f"  WARNING: could not load CESM2: {e}")
        cesm_years, cesm_mean = None, None

    # ── compute baselines and anomalies ───────────────────────────────────
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    w /= w.mean()

    def gmean_anom(ens, yrs):
        """(N, T, H, W) → (N, T) anomalies; baseline from 1850-1900."""
        gmean = (ens * w).mean(axis=(-2, -1))         # (N, T)
        mask  = (yrs >= BASELINE_START) & (yrs <= BASELINE_END)
        bl    = gmean[:, mask].mean(axis=1, keepdims=True)  # (N, 1)
        return gmean - bl

    anom_normal = gmean_anom(ens_normal, years)   # (N, T)
    anom_nosul  = gmean_anom(ens_nosul,  years)

    if cesm_mean is not None:
        gmean_cesm = (cesm_mean * w).mean(axis=(-2, -1))   # (T,)
        mask_bl    = (cesm_years >= BASELINE_START) & (cesm_years <= BASELINE_END)
        anom_cesm  = gmean_cesm - gmean_cesm[mask_bl].mean()

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: anomaly time series
    ax = axes[0]
    for m in range(args.n_ensemble):
        ax.plot(years, anom_normal[m], color="royalblue",  lw=0.7, alpha=0.4)
        ax.plot(years, anom_nosul[m],  color="tomato",     lw=0.7, alpha=0.4)
    ax.plot(years, anom_normal.mean(0), color="royalblue", lw=2,
            label="Model – normal SUL")
    ax.plot(years, anom_nosul.mean(0),  color="tomato",    lw=2,
            label="Model – SUL = minimum (no aerosols)")
    if cesm_mean is not None:
        ax.plot(cesm_years, anom_cesm, color="k", lw=2, ls="--",
                label="CESM2 hist ensemble mean")
    ax.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12,
               label="baseline 1850-1900")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("TREFHT anomaly (°C)")
    ax.set_title("Hist: normal vs no-aerosol (SUL=min) conditioning")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 2: NoSUL − Normal (= model's aerosol response)
    ax = axes[1]
    diff_members = anom_nosul - anom_normal    # (N, T)
    ax.fill_between(years, diff_members.min(0), diff_members.max(0),
                    color="purple", alpha=0.2, label="member spread")
    ax.plot(years, diff_members.mean(0), color="purple", lw=2,
            label="NoSUL − Normal (= model aerosol warming effect)")
    ax.axhline(0, color="k", lw=0.9)
    ax.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12)
    ax.set_ylabel("ΔT (°C)  [NoSUL − Normal]")
    ax.set_title("Model aerosol response: warming when aerosols removed\n"
                 "(positive = model IS using SUL; magnitude shows how much)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 3: Model bias vs CESM2 (normal conditioning)
    ax = axes[2]
    if cesm_mean is not None:
        common, ig, ic = np.intersect1d(years, cesm_years, return_indices=True)
        bias = anom_normal.mean(0)[ig] - anom_cesm[ic]
        ax.plot(common, bias, color="royalblue", lw=2,
                label="Model bias (Normal − CESM2)")
        ax.fill_between(common,
                        anom_normal[:, ig].min(0) - anom_cesm[ic],
                        anom_normal[:, ig].max(0) - anom_cesm[ic],
                        color="royalblue", alpha=0.15)
        # Also show how much of the bias is explained by aerosol under-response
        aerosol_effect = diff_members.mean(0)[ig]   # model aerosol warming effect
        ax.plot(common, aerosol_effect, color="purple", lw=1.5, ls="--",
                label="Model aerosol effect (NoSUL−Normal)\n≈ missing aerosol cooling if bias matches")
        ax.axhline(0, color="k", lw=0.9)
        ax.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12)
        ax.set_ylabel("°C")
        ax.set_title("Model bias vs CESM2 vs aerosol under-response\n"
                     "(if purple ≈ blue → aerosol under-response explains the warm bias)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "CESM2 data not available", ha="center",
                transform=ax.transAxes)

    ax.set_xlabel("Year")

    # Print key numbers
    print("\n[RESULTS] Global-mean aerosol response (NoSUL − Normal):")
    diff_mean = diff_members.mean(0)
    for yr_target in [1900, 1950, 1970, 1980, 2000, 2014]:
        idx = np.argmin(np.abs(years - yr_target))
        print(f"  {years[idx]}: {diff_mean[idx]:+.3f} °C")

    if cesm_mean is not None:
        print("\n[RESULTS] Model warm bias (Normal − CESM2):")
        for yr_target in [1900, 1950, 1970, 1980, 2000, 2014]:
            idx_m = np.argmin(np.abs(years    - yr_target))
            idx_c = np.argmin(np.abs(cesm_years - yr_target))
            b = anom_normal.mean(0)[idx_m] - anom_cesm[idx_c]
            print(f"  {years[idx_m]}: {b:+.3f} °C")

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "aerosol_sensitivity_hist.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
