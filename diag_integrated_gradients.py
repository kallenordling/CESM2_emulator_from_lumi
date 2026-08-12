#!/usr/bin/env python3
"""diag_integrated_gradients.py
================================
Explainability diagnostic: Integrated Gradients over CO2 and SUL conditioning.

For each time step in a scenario, computes how much the CO2 and SUL conditioning
values contribute to the model's predicted global-mean temperature.

Method: Integrated Gradients (Sundararajan et al. 2017)
  - Baseline: null conditioning (−1.0) = CFG null = pre-industrial state
  - Target:   actual CO2 / SUL conditioning at each time step
  - Attribution ≈ (actual − null) × mean gradient over N interpolation steps

A single differentiable denoising step at low noise (t_proxy) is used as a
fast proxy for the full generation.  Gradients are accumulated over N_IG_STEPS
interpolations between null and actual conditioning.

Output panels:
  1. CO2 attribution over time (hist + ssp370)
  2. SUL attribution over time
  3. Relative CO2 fraction |CO2| / (|CO2| + |SUL|)

If panel 3 collapses to 0 after 2060 in ssp370 → CO2-aerosol collinearity confirmed.

Run on LUMI:
    python diag_integrated_gradients.py [--checkpoint /path/to/run.pt]
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

# ── paths ──────────────────────────────────────────────────────────────────────
PROJ_ROOT   = f"{L.REPO}"
SCRATCH     = f"{L.DATA}"
RUNS_DIR    = os.path.join(PROJ_ROOT, "runs")
EMIS_DIR    = SCRATCH
CONFIG_PATH = "configs/config_aero.yaml"
COND_VARS   = ["CO2", "SUL"]
NULL_COND   = -1.0

# Scenarios to analyse and their emission files
SCENARIOS = [
    ("hist",   os.path.join(EMIS_DIR, "emissions_hist_timefixed.nc")),
    ("ssp370", os.path.join(EMIS_DIR, "emissions_ssp370_timefixed.nc")),
]

# Physical scale factor for normalised → °C conversion (from training normalisation)
T_SCALE = 21.0   # model output is (T_K - 277.65) / 21.0 approx


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


def load_model(ckpt_path: str, config_path: str, device):
    cfg   = OmegaConf.load(config_path)
    model = instantiate(cfg.model)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["EMA"], strict=False)
    model = model.to(device).eval()
    # Disable parameter gradients — only cond_map gradients are needed for IG
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_years(coord_vals) -> np.ndarray:
    if hasattr(coord_vals[0], "year"):
        return np.array([int(str(v)[:4]) for v in coord_vals])
    return np.asarray(coord_vals, dtype=int)


def build_cond_tensor(cond_file: str, time_dim: str = "time"):
    """Return (cond_tensor (2, T, H, W), years (T,), lat (H,))."""
    raw     = xr.open_dataset(cond_file, chunks={time_dim: -1})[COND_VARS]
    norm    = raw.map(normalize)
    lat     = norm["lat"].values.astype(np.float64)
    stacked = norm.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"])
    stacked = stacked.transpose("var", time_dim, "lat", "lon")
    tensor  = torch.tensor(stacked.values, dtype=torch.float32)  # (2, T, H, W)
    years   = extract_years(norm[time_dim].values)
    raw.close()
    return tensor, years, lat


def compute_integrated_gradients(model, scheduler, cond_tensor: torch.Tensor,
                                  lat: np.ndarray, device, dtype,
                                  n_ig_steps: int = 50, batch_size: int = 32,
                                  t_proxy: float = 0.2, seed: int = 42):
    """Compute Integrated Gradients of global-mean T w.r.t. CO2 and SUL.

    Each time step is processed independently (the model has no temporal memory
    across time steps — conditioning is applied per-step).

    A single denoising step at t=t_proxy provides a differentiable proxy for
    the full generation.  Gradients are accumulated over n_ig_steps interpolation
    steps between the null baseline (−1.0) and the actual conditioning.

    Returns
    -------
    ig_co2  : (T,)  Integrated Gradient attribution for CO2 channel (°C-norm)
    ig_sul  : (T,)  Integrated Gradient attribution for SUL channel (°C-norm)
    d_co2   : (T,)  CO2 conditioning delta from null (actual − null)
    d_sul   : (T,)  SUL conditioning delta from null
    """
    _, T_total, H, W = cond_tensor.shape

    # Area weights: cos(lat), normalised to mean=1, shape (H, 1)
    w_lat = torch.tensor(np.cos(np.deg2rad(lat)), dtype=dtype, device=device)
    w_lat = (w_lat / w_lat.mean()).view(H, 1)

    ig_co2 = np.zeros(T_total, dtype=np.float32)
    ig_sul = np.zeros(T_total, dtype=np.float32)
    d_co2  = np.zeros(T_total, dtype=np.float32)
    d_sul  = np.zeros(T_total, dtype=np.float32)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for t_start in tqdm(range(0, T_total, batch_size), desc="  IG batches"):
        t_end = min(t_start + batch_size, T_total)
        B = t_end - t_start

        # Actual conditioning for this chunk: (B, 2, 1, H, W)
        cond_actual = (
            cond_tensor[:, t_start:t_end]   # (2, B, H, W)
            .permute(1, 0, 2, 3)            # (B, 2, H, W)
            .unsqueeze(2)                   # (B, 2, 1, H, W)
            .to(device=device, dtype=dtype)
        )
        cond_null_b = torch.full_like(cond_actual, NULL_COND)
        delta       = cond_actual - cond_null_b   # (B, 2, 1, H, W)

        # Fixed noise — same across all IG interpolation steps so the only thing
        # that varies is the conditioning, giving a clean gradient estimate.
        noise = torch.randn(B, 1, 1, H, W, device=device, dtype=dtype,
                            generator=rng)

        # Single-step denoising proxy at t=t_proxy (moderate noise level).
        # x_clean = 0 so x_noisy is pure noise scaled by the noise schedule;
        # the model's conditioned prediction relative to this blank canvas is
        # what we differentiate.
        t_val     = torch.full((B,), t_proxy, device=device, dtype=dtype)
        log_snr   = scheduler.log_snr(t_val)
        x_clean   = torch.zeros(B, 1, 1, H, W, device=device, dtype=dtype)
        x_noisy   = scheduler.add_noise(x_clean, noise, log_snr).detach()
        log_snr   = log_snr.detach()

        # Accumulate gradients over interpolation steps
        sum_grad_co2 = torch.zeros(B, device=device, dtype=dtype)
        sum_grad_sul = torch.zeros(B, device=device, dtype=dtype)

        for k in range(1, n_ig_steps + 1):
            alpha = k / n_ig_steps

            # Leaf tensor at this interpolation point — gradients land here
            cond_k = (cond_null_b + alpha * delta).detach().requires_grad_(True)

            # Forward: v-prediction → decode to x0
            v_pred  = model(x_noisy, log_snr, cond_map=cond_k)
            pred_x0 = scheduler.predict_start_from_v(x_noisy, log_snr, v_pred)
            # pred_x0: (B, 1, 1, H, W) in normalised temperature units

            # Area-weighted global mean T (scalar per batch element)
            pred_map = pred_x0.squeeze(1).squeeze(1)            # (B, H, W)
            T_global = (pred_map * w_lat).mean(dim=(-2, -1))    # (B,)

            # Gradient of summed T_global w.r.t. cond_k: shape (B, 2, 1, H, W)
            grads = torch.autograd.grad(T_global.sum(), cond_k)[0]

            # Spatially-averaged gradient per channel (CO2 ≈ uniform, SUL varies)
            sum_grad_co2 += grads[:, 0].mean(dim=(-2, -1)).squeeze(1)   # (B,)
            sum_grad_sul += grads[:, 1].mean(dim=(-2, -1)).squeeze(1)

        # IG = (actual − null) × mean_gradient  [Riemann sum approximation]
        d_co2_b = delta[:, 0].mean(dim=(-2, -1)).squeeze(1).detach().cpu().numpy()
        d_sul_b = delta[:, 1].mean(dim=(-2, -1)).squeeze(1).detach().cpu().numpy()

        mean_g_co2 = (sum_grad_co2 / n_ig_steps).cpu().numpy()
        mean_g_sul = (sum_grad_sul / n_ig_steps).cpu().numpy()

        ig_co2[t_start:t_end] = d_co2_b * mean_g_co2
        ig_sul[t_start:t_end] = d_sul_b * mean_g_sul
        d_co2[t_start:t_end]  = d_co2_b
        d_sul[t_start:t_end]  = d_sul_b

    return ig_co2, ig_sul, d_co2, d_sul


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",    default=RUNS_DIR)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--output-dir",  default="${LUMI_EVAL_OUT}")
    parser.add_argument("--n-ig-steps",  type=int,   default=50,
                        help="Number of IG interpolation steps (more = more accurate, slower)")
    parser.add_argument("--batch-size",  type=int,   default=32,
                        help="Time steps per GPU batch")
    parser.add_argument("--t-proxy",     type=float, default=0.2,
                        help="Noise level for single-step denoising proxy (0=clean, 1=pure noise)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    print(f"[DEVICE] {device}")
    print(f"[IG] n_ig_steps={args.n_ig_steps}  batch_size={args.batch_size}"
          f"  t_proxy={args.t_proxy}")

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.runs_dir)
    model     = load_model(ckpt_path, CONFIG_PATH, device).to(dtype)

    cfg       = OmegaConf.load(CONFIG_PATH)
    scheduler = instantiate(cfg.scheduler)

    # ── run IG for each scenario ───────────────────────────────────────────────
    results = {}
    for scen_name, cond_file in SCENARIOS:
        if not os.path.exists(cond_file):
            print(f"[SKIP] {scen_name}: {cond_file} not found")
            continue
        print(f"\n[IG] {scen_name} ({cond_file}) …")
        cond_t, years, lat = build_cond_tensor(cond_file)
        ig_co2, ig_sul, d_co2, d_sul = compute_integrated_gradients(
            model, scheduler, cond_t, lat, device, dtype,
            n_ig_steps=args.n_ig_steps,
            batch_size=args.batch_size,
            t_proxy=args.t_proxy,
        )
        results[scen_name] = dict(years=years, ig_co2=ig_co2, ig_sul=ig_sul,
                                   d_co2=d_co2, d_sul=d_sul)

    if not results:
        print("[ERROR] No scenarios loaded — check EMIS_DIR paths.")
        return

    # ── print results ──────────────────────────────────────────────────────────
    KEY_YEARS = [1900, 1950, 1970, 1980, 2000, 2010, 2014,
                 2020, 2030, 2050, 2060, 2070, 2080, 2090, 2100]
    for scen_name, r in results.items():
        print(f"\n[RESULTS] Integrated Gradients for {scen_name}:")
        print(f"  {'Year':>6}  {'CO2_attr':>10}  {'SUL_attr':>10}  "
              f"{'|CO2|/(|CO2|+|SUL|)':>22}  {'Δcond_CO2':>10}  {'Δcond_SUL':>10}")
        for yr in KEY_YEARS:
            idx_arr = np.where(r["years"] == yr)[0]
            if len(idx_arr) == 0:
                continue
            idx  = idx_arr[0]
            co2a = float(r["ig_co2"][idx])
            sula = float(r["ig_sul"][idx])
            frac = abs(co2a) / (abs(co2a) + abs(sula) + 1e-10)
            dc   = float(r["d_co2"][idx])
            ds   = float(r["d_sul"][idx])
            print(f"  {yr:>6}  {co2a:>+10.4f}  {sula:>+10.4f}  "
                  f"{frac:>22.2%}  {dc:>+10.4f}  {ds:>+10.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    COLORS = {"hist": "steelblue", "ssp370": "tomato"}

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)

    # Panel 1: CO2 attribution
    ax = axes[0]
    for name, r in results.items():
        ax.plot(r["years"], r["ig_co2"] * T_SCALE,
                color=COLORS.get(name, "grey"), lw=1.8, label=name)
    ax.axvline(2014, color="k", lw=0.8, ls="--", alpha=0.5, label="hist / ssp370")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("CO2 IG attribution (°C-equiv)")
    ax.set_title("Integrated Gradients: CO2 channel\n"
                 "(positive = CO2 drives warming; collapse → model ignores CO2)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 2: SUL attribution
    ax = axes[1]
    for name, r in results.items():
        ax.plot(r["years"], r["ig_sul"] * T_SCALE,
                color=COLORS.get(name, "grey"), lw=1.8, label=name)
    ax.axvline(2014, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("SUL IG attribution (°C-equiv)")
    ax.set_title("Integrated Gradients: SUL (aerosol) channel\n"
                 "(negative = aerosols cool; if near-zero post-2050 → model stopped using SUL)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 3: relative CO2 fraction
    ax = axes[2]
    for name, r in results.items():
        denom = np.abs(r["ig_co2"]) + np.abs(r["ig_sul"]) + 1e-10
        frac  = np.abs(r["ig_co2"]) / denom
        ax.plot(r["years"], frac, color=COLORS.get(name, "grey"),
                lw=1.8, label=name)
    ax.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.6, label="50 / 50")
    ax.axvline(2014, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("|CO2_attr| / (|CO2_attr| + |SUL_attr|)")
    ax.set_title("Relative CO2 attribution\n"
                 "(1.0 = model driven only by CO2;  0.0 = only by SUL;\n"
                 " collapse to 0 after ~2060 in ssp370 = CO2-aerosol collinearity problem)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Year")

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "integrated_gradients.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
