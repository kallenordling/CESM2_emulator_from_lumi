#!/usr/bin/env python3
"""diag_co2_sensitivity.py
===========================
Diagnostic: is the model's CO2 response too strong?

Runs ssp370 conditioning (2015–2100, CO2 up to ~1000 ppm) THREE ways:
  1. Normal  — CO2 and SUL as observed
  2. NoCO2   — CO2 pinned to its minimum (-1, i.e. pre-industrial ~285 ppm),
               SUL as observed
  3. CESM2   — actual ssp370 ensemble mean (reference)

The difference (Normal − NoCO2) is the model's CO2-forced warming response.

We then compare this to:
  a) CESM2 ssp370 actual warming (what the CO2 response should be)
  b) The theoretical logarithmic forcing curve: ΔT = (ECS/ln2) × ln(CO2/CO2_ref)
     with CESM2's ECS ≈ 3.6°C and CO2_ref = 285 ppm

If the model curve lies above the logarithmic curve (or above CESM2), the CO2
channel is too strong — explaining the warm bias in ssp370/ghg at high CO2.

Run on LUMI:
    python diag_co2_sensitivity.py [--checkpoint /path/to/run.pt]
"""

import lumi_paths as L
import argparse
import glob
import os
import re

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from data.climate_dataset import normalize
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from models.video_net import UNetModel3D

# ── paths ──────────────────────────────────────────────────────────────────────
PROJ_ROOT   = f"{L.REPO}"
SCRATCH     = f"{L.DATA}"
RUNS_DIR    = os.path.join(PROJ_ROOT, "runs")
DATA_ROOT   = os.path.join(SCRATCH, "training_data/TREFHT")
EMIS_DIR    = SCRATCH
CONFIG_PATH = "configs/config_aero.yaml"

SSP370_DATA_DIR  = os.path.join(DATA_ROOT, "ssp370")
SSP370_COND_FILE = os.path.join(EMIS_DIR, "emissions_ssp370_timefixed.nc")
SSP370_MEMBERS   = ["LE2-1001.001", "LE2-1011.001", "LE2-1021.002",
                    "LE2-1031.002", "LE2-1041.003"]

# Also load hist for baseline computation
HIST_DATA_DIR  = os.path.join(DATA_ROOT, "hist")
HIST_COND_FILE = os.path.join(EMIS_DIR, "emissions_hist_timefixed.nc")
HIST_MEMBERS   = ["LE2-1001.001", "LE2-1011.001", "LE2-1021.002",
                  "LE2-1031.002", "LE2-1041.003"]

BASELINE_START = 1850
BASELINE_END   = 1900
SAMPLE_STEPS   = 50
BATCH_SIZE     = 16
N_ENSEMBLE     = 3
COND_VARS      = ["CO2", "SUL"]
CO2_IDX        = 0
SUL_IDX        = 1

# CESM2 ECS ≈ 3.6°C per doubling (used for logarithmic reference curve)
CESM2_ECS      = 3.6
CO2_PREIND     = 285.0   # ppm, approximate 1850 value


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
    cfg   = OmegaConf.load(config_path)
    model = instantiate(cfg.model)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["EMA"], strict=False)
    return model.to(device).eval(), ckpt.get("PCA")


def extract_years(coord_vals) -> np.ndarray:
    if hasattr(coord_vals[0], "year"):
        return np.array([int(str(v)[:4]) for v in coord_vals])
    return np.asarray(coord_vals, dtype=int)


def build_cond_tensor(cond_file, time_dim="time"):
    """Return (cond_tensor (C, T, H, W), years, lat, lon)."""
    raw     = xr.open_dataset(cond_file, chunks={time_dim: -1})[COND_VARS]
    norm    = raw.map(normalize)
    lat     = norm["lat"].values.astype(np.float64)
    lon     = norm["lon"].values.astype(np.float64)
    stacked = norm.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"])
    stacked = stacked.transpose("var", time_dim, "lat", "lon")
    tensor  = torch.tensor(stacked.values, dtype=torch.float32)
    years   = extract_years(norm[time_dim].values)
    raw.close()
    return tensor, years, lat, lon


def generate_ensemble(model, scheduler, cond_tensor, device, dtype,
                      n_ensemble, sample_steps, batch_size, label=""):
    """Generate N ensemble members → (N, T, H, W) in °C."""
    members = []
    for m in range(n_ensemble):
        print(f"  {label} member {m+1}/{n_ensemble}")
        torch.manual_seed(m)
        _, T, H, W = cond_tensor.shape
        scheduler.set_timesteps(sample_steps)
        steps   = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        results = []
        for i in tqdm(range(0, T, batch_size), desc="    batches", leave=False):
            chunk  = cond_tensor[:, i:i+batch_size]
            B      = chunk.shape[1]
            cond_b = chunk.permute(1, 0, 2, 3).unsqueeze(2).to(device=device, dtype=dtype)
            gen    = torch.randn(B, 1, 1, H, W, device=device, dtype=dtype)
            with torch.no_grad():
                for _, t_idx in enumerate(scheduler.timesteps):
                    t    = scheduler.log_snr(steps[t_idx]).expand(B).to(dtype)
                    pred = model(gen, t, cond_map=cond_b)
                    gen  = scheduler.step(pred, timestep=t_idx, sample=gen).prev_sample
            results.append(gen.squeeze(1).squeeze(1).cpu().float())
        gen_norm = torch.cat(results, dim=0).numpy()
        members.append(gen_norm * 21.0 + 4.5)
    return np.stack(members, axis=0)   # (N, T, H, W)


def area_weighted_gmean(data: np.ndarray, lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    w /= w.mean()
    return (data * w).mean(axis=(-2, -1))


def load_cesm2_ensemble(data_dir, realizations, time_dim="time"):
    members, common_years = [], None
    for real in realizations:
        path = os.path.join(data_dir, real, "*.nc")
        ds   = xr.open_mfdataset(path, combine="by_coords",
                                  chunks={time_dim: 50})["TREFHT"] - 273.15
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


def co2_ppm_from_norm(co2_norm_tensor: torch.Tensor,
                      co2_min_ppm: float, co2_max_ppm: float) -> np.ndarray:
    """Convert normalised CO2 back to ppm for the logarithmic reference curve.
    norm = (ppm - mean) / (range/2)  →  ppm = norm * (range/2) + mean
    """
    mean_val  = (co2_min_ppm + co2_max_ppm) / 2.0
    range_val = co2_max_ppm - co2_min_ppm
    return co2_norm_tensor.numpy() * (range_val / 2.0) + mean_val


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",     default=RUNS_DIR)
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--output-dir",   default=f"{L.EVAL_OUT}")
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

    # ── conditioning tensors ───────────────────────────────────────────────────
    print("\n[COND] Building ssp370 conditioning …")
    cond_normal, ssp_years, lat, lon = build_cond_tensor(SSP370_COND_FILE,
                                                          time_dim="time")

    co2_min_norm = float(cond_normal[CO2_IDX].min())
    co2_max_norm = float(cond_normal[CO2_IDX].max())
    print(f"  ssp370 CO2 (normalised): min={co2_min_norm:.3f}  max={co2_max_norm:.3f}")
    print(f"  ssp370 SUL (normalised): min={float(cond_normal[SUL_IDX].min()):.3f}"
          f"  max={float(cond_normal[SUL_IDX].max()):.3f}")

    # NoCO2: CO2 pinned to global minimum, SUL unchanged
    cond_noco2 = cond_normal.clone()
    cond_noco2[CO2_IDX] = co2_min_norm
    print(f"  NoCO2: CO2 fixed at {co2_min_norm:.3f} (pre-industrial minimum)")

    # Estimate actual ppm range from hist+ssp370 combined
    # (approximation — used only for the logarithmic reference curve)
    print("\n[COND] Loading hist CO2 to estimate ppm range for log curve …")
    try:
        raw_hist   = xr.open_dataset(HIST_COND_FILE)["CO2"]
        raw_ssp    = xr.open_dataset(SSP370_COND_FILE)["CO2"]
        co2_ppm_lo = float(min(raw_hist.min(), raw_ssp.min()))
        co2_ppm_hi = float(max(raw_hist.max(), raw_ssp.max()))
        raw_hist.close(); raw_ssp.close()
        print(f"  CO2 ppm range: {co2_ppm_lo:.1f} – {co2_ppm_hi:.1f} ppm")
    except Exception as e:
        print(f"  WARNING: could not read raw CO2 ppm ({e}). "
              f"Using defaults: {CO2_PREIND}–1000 ppm")
        co2_ppm_lo, co2_ppm_hi = CO2_PREIND, 1000.0

    # ── generate ──────────────────────────────────────────────────────────────
    print("\n[GEN] ssp370 Normal conditioning …")
    ens_normal = generate_ensemble(model, scheduler, cond_normal, device, dtype,
                                   args.n_ensemble, args.sample_steps,
                                   BATCH_SIZE, label="Normal")

    print("\n[GEN] ssp370 NoCO2 conditioning (CO2 = pre-industrial minimum) …")
    ens_noco2  = generate_ensemble(model, scheduler, cond_noco2, device, dtype,
                                   args.n_ensemble, args.sample_steps,
                                   BATCH_SIZE, label="NoCO2")

    # ── CESM2 ssp370 + hist (for baseline) ────────────────────────────────────
    print("\n[CESM2] Loading ssp370 ensemble …")
    try:
        cesm_ssp_years, cesm_ssp_ens = load_cesm2_ensemble(
            SSP370_DATA_DIR, SSP370_MEMBERS)
        cesm_ssp_mean = cesm_ssp_ens.mean(axis=0)
    except Exception as e:
        print(f"  WARNING: {e}")
        cesm_ssp_years, cesm_ssp_mean = None, None

    print("\n[CESM2] Loading hist ensemble for 1850-1900 baseline …")
    try:
        cesm_hist_years, cesm_hist_ens = load_cesm2_ensemble(
            HIST_DATA_DIR, HIST_MEMBERS)
        cesm_hist_mean = cesm_hist_ens.mean(axis=0)
        mask_bl    = ((cesm_hist_years >= BASELINE_START) &
                      (cesm_hist_years <= BASELINE_END))
        baseline_map = cesm_hist_mean[mask_bl].mean(axis=0)   # (H, W)
        print(f"  Baseline map: mean={baseline_map.mean():.2f}°C")
    except Exception as e:
        print(f"  WARNING: could not load hist for baseline ({e}). "
              f"Using model 1850-1900 mean instead.")
        baseline_map = None

    # ── area-weighted global means ─────────────────────────────────────────────
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    w /= w.mean()

    def gmean_anom_ens(ens, yrs, bmap):
        """(N, T, H, W) → (N, T) global-mean anomaly re 1850-1900."""
        gmean = (ens * w).mean(axis=(-2, -1))      # (N, T)
        if bmap is not None:
            bl = float((bmap * w).mean())
        else:
            mask = (yrs >= BASELINE_START) & (yrs <= BASELINE_END)
            bl   = gmean[:, mask].mean()
        return gmean - bl

    anom_normal = gmean_anom_ens(ens_normal, ssp_years, baseline_map)
    anom_noco2  = gmean_anom_ens(ens_noco2,  ssp_years, baseline_map)

    if cesm_ssp_mean is not None and baseline_map is not None:
        gmean_cesm  = (cesm_ssp_mean * w).mean(axis=(-2, -1))
        bl_scalar   = float((baseline_map * w).mean())
        anom_cesm   = gmean_cesm - bl_scalar
    else:
        anom_cesm = None

    # ── CO2 response and logarithmic reference ────────────────────────────────
    # Model CO2 response per year: Normal − NoCO2 (positive = warming from CO2)
    co2_response = anom_normal - anom_noco2    # (N, T)

    # Logarithmic reference: ΔT = (ECS / ln2) × ln(CO2_t / CO2_ref)
    # Use the spatial-mean normalised CO2 from ssp370 conditioning
    co2_norm_ts   = cond_normal[CO2_IDX].mean(dim=(-2, -1)).cpu()   # (T,)
    co2_ppm_ts    = co2_ppm_from_norm(co2_norm_ts, co2_ppm_lo, co2_ppm_hi)
    co2_ppm_ts    = np.clip(co2_ppm_ts, CO2_PREIND, None)
    log_ref_dt    = (CESM2_ECS / np.log(2)) * np.log(co2_ppm_ts / CO2_PREIND)

    # ── print results ─────────────────────────────────────────────────────────
    print("\n[RESULTS] Model CO2 response (Normal − NoCO2) for ssp370:")
    for yr_target in [2020, 2030, 2050, 2070, 2090, 2100]:
        idx = np.argmin(np.abs(ssp_years - yr_target))
        r   = co2_response.mean(0)[idx]
        lr  = log_ref_dt[idx]
        print(f"  {ssp_years[idx]}: model={r:+.3f} °C   log_ref={lr:+.3f} °C"
              f"   ratio={r/lr:.2f}x" if lr > 0 else f"  {ssp_years[idx]}: model={r:+.3f} °C")

    if anom_cesm is not None:
        print("\n[RESULTS] Model warm bias (Normal − CESM2) for ssp370:")
        for yr_target in [2020, 2030, 2050, 2070, 2090, 2100]:
            idx_m = np.argmin(np.abs(ssp_years       - yr_target))
            idx_c = np.argmin(np.abs(cesm_ssp_years  - yr_target))
            bias  = anom_normal.mean(0)[idx_m] - anom_cesm[idx_c]
            print(f"  {ssp_years[idx_m]}: {bias:+.3f} °C")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # Panel 1: anomaly time series
    ax = axes[0]
    for m in range(args.n_ensemble):
        ax.plot(ssp_years, anom_normal[m], color="royalblue", lw=0.7, alpha=0.4)
        ax.plot(ssp_years, anom_noco2[m],  color="grey",      lw=0.7, alpha=0.4)
    ax.plot(ssp_years, anom_normal.mean(0), color="royalblue", lw=2,
            label="Model – normal conditioning")
    ax.plot(ssp_years, anom_noco2.mean(0),  color="grey",      lw=2,
            label="Model – CO2 = pre-industrial (NoCO2)")
    if anom_cesm is not None:
        ax.plot(cesm_ssp_years, anom_cesm, color="k", lw=2, ls="--",
                label="CESM2 ssp370 ensemble mean")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("TREFHT anomaly (°C)")
    ax.set_title("ssp370: normal vs no-CO2 conditioning (re 1850–1900)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 2: CO2 response vs logarithmic reference
    ax = axes[1]
    ax.fill_between(ssp_years,
                    co2_response.min(0), co2_response.max(0),
                    color="royalblue", alpha=0.2, label="member spread")
    ax.plot(ssp_years, co2_response.mean(0), color="royalblue", lw=2,
            label="Model CO2 response (Normal − NoCO2)")
    ax.plot(ssp_years, log_ref_dt, color="orange", lw=2, ls="--",
            label=f"Logarithmic reference (ECS={CESM2_ECS}°C, CO2_ref={CO2_PREIND:.0f} ppm)")
    if anom_cesm is not None:
        # CESM2 total warming = approximate upper bound on CO2 response
        common, ig, ic = np.intersect1d(ssp_years, cesm_ssp_years,
                                         return_indices=True)
        ax.plot(common, anom_cesm[ic], color="k", lw=1.5, ls=":",
                label="CESM2 total anomaly (includes aerosol+CO2)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("ΔT (°C)")
    ax.set_title("Model CO2 response vs logarithmic forcing reference\n"
                 "(if blue >> orange → CO2 channel too strong)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 3: model bias vs CO2 over-response
    ax = axes[2]
    if anom_cesm is not None:
        common, ig, ic = np.intersect1d(ssp_years, cesm_ssp_years,
                                         return_indices=True)
        bias      = anom_normal.mean(0)[ig] - anom_cesm[ic]
        co2_over  = co2_response.mean(0)[ig] - log_ref_dt[ig]  # excess above log ref
        ax.plot(common, bias,     color="royalblue", lw=2,
                label="Warm bias (Normal − CESM2)")
        ax.fill_between(common,
                        anom_normal[:, ig].min(0) - anom_cesm[ic],
                        anom_normal[:, ig].max(0) - anom_cesm[ic],
                        color="royalblue", alpha=0.15)
        ax.plot(common, co2_over, color="orange", lw=2, ls="--",
                label="CO2 excess response (model − log reference)\n"
                      "≈ warm bias explained by CO2 over-response if matches blue")
        ax.axhline(0, color="k", lw=0.9)
        ax.set_ylabel("°C")
        ax.set_title("Does CO2 over-response explain the warm bias?\n"
                     "(orange ≈ blue → yes; orange ≪ blue → other causes)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "CESM2 data not available", ha="center",
                transform=ax.transAxes)

    ax.set_xlabel("Year")
    fig.tight_layout()

    out_path = os.path.join(args.output_dir, "co2_sensitivity_ssp370.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
