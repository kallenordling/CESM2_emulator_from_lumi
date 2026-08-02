"""Standalone diagnostic (hypothesis #3 of the ssp370 warm+wet bias investigation,
2026-07-22): the SGAIN/TCRE-slope target that constrains ssp370's sensitivity is
fit in trainer/unetTrainer.py `_precompute_tcre_slope` -> `_gmean_trajectory`
using ONLY the first realization in each scenario's realization list (see
`ds.load_data(first_real)` there), not an ensemble mean. If ssp370's true
per-realization slope varies a lot around that single draw, the SGAIN target
itself could be a noisy/biased anchor -- independent of the two A/B probes
(batch-share, interaction-match) already run and ruled out (see memory
gainfix_ssp370_persistent_bias.md).

This script recomputes the same area-weighted gmean(dT) vs gmean(cumCO2) slope
fit, but looping over EVERY realization for hist and ssp370, and compares:
  - the single-realization slope (matches what training actually uses)
  - the per-realization slope distribution (mean/std across realizations)
  - the pooled-data slope (all realizations concatenated, one fit)

No SLURM job needed -- reads training data directly, run on a login node.

Usage (on LUMI):
    python diag_sgain_slope_ensemble.py [--data-config configs/config_data.yaml]
"""
import argparse
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")
from data.climate_dataset import ClimateDataset  # noqa: E402


def area_weights(lats: np.ndarray) -> torch.Tensor:
    lats_t = torch.as_tensor(lats, dtype=torch.float32)
    w = torch.cos(torch.deg2rad(lats_t)).clamp(min=0.2)
    w = w / w.mean()
    return w.view(1, -1, 1)  # (1, H, 1)


def gmean_trajectory(ds: ClimateDataset, realization: str, clim_t: torch.Tensor,
                      w_lat: torch.Tensor):
    ds.load_data(realization)
    t = ds.tensor_data        # (n_vars_target, T, H, W)
    c = ds.tensor_data_cond   # (n_vars_cond,   T, H, W) -- ch 0 = CO2
    dT = t[0] - clim_t.unsqueeze(0)
    co2 = c[0]
    dT_gm = (dT * w_lat).mean(dim=(1, 2)).to(torch.float64).numpy()
    co2_gm = (co2 * w_lat).mean(dim=(1, 2)).to(torch.float64).numpy()
    return co2_gm, dT_gm


def build_dataset(cfg: dict, scen_cfg: dict, external_climatology=None) -> ClimateDataset:
    return ClimateDataset(
        seq_len=cfg["seq_len"],
        realizations=scen_cfg["realizations"],
        data_dir=scen_cfg["data_dir"],
        target_vars=cfg["target_vars"],
        cond_file=scen_cfg["cond_file"],
        cond_vars=cfg["cond_vars"],
        n_components_target=cfg.get("n_components_target"),
        n_components_cond=cfg.get("n_components_cond"),
        cond_smooth_sigma=cfg.get("cond_smooth_sigma"),
        cond_smooth_method=cfg.get("cond_smooth_method", "gaussian"),
        time_dim=scen_cfg.get("time_dim", "year"),
        external_climatology=external_climatology,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/config_data.yaml")
    args = ap.parse_args()

    with open(args.data_config) as f:
        cfg = yaml.safe_load(f)

    exp_by_name = {e["scenario_name"]: e for e in cfg["experiment_configs"]}
    hist_cfg = exp_by_name["hist"]
    ssp370_cfg = exp_by_name["ssp370"]

    print(f"[diag] hist realizations ({len(hist_cfg['realizations'])}): "
          f"{hist_cfg['realizations']}")
    print(f"[diag] ssp370 realizations ({len(ssp370_cfg['realizations'])}): "
          f"{ssp370_cfg['realizations']}")

    # ── hist: build with first realization only, matching how the trainer's
    # multi_experiment_dataset builder derives hist_climatology (ClimateDataset
    # itself only loads self.realizations[0] in __init__).
    hist_ds = build_dataset(cfg, {**hist_cfg, "realizations": hist_cfg["realizations"]})
    clim = hist_ds.climatology.detach().squeeze(0).squeeze(1)  # (C, H, W)
    clim_t = clim[0]
    w_lat = area_weights(hist_ds.lats.values)

    ssp370_ds = build_dataset(cfg, ssp370_cfg, external_climatology=hist_ds.climatology)

    # ── Single-realization slope, EXACTLY as training computes it ───────────
    x_h0, y_h0 = gmean_trajectory(hist_ds, hist_cfg["realizations"][0], clim_t, w_lat)
    x_s0, y_s0 = gmean_trajectory(ssp370_ds, ssp370_cfg["realizations"][0], clim_t, w_lat)
    slope_h0, _ = np.polyfit(x_h0, y_h0, 1)
    slope_s0, _ = np.polyfit(x_s0, y_s0, 1)
    print(f"\n[SINGLE-REALIZATION, matches training _precompute_tcre_slope]")
    print(f"  hist   (real={hist_cfg['realizations'][0]}):   slope={slope_h0:.4f}")
    print(f"  ssp370 (real={ssp370_cfg['realizations'][0]}): slope={slope_s0:.4f}")

    # ── Per-realization slope distribution + pooled fit ──────────────────────
    def scan_all(ds: ClimateDataset, scen_cfg: dict):
        slopes = []
        xs_all, ys_all = [], []
        for real in scen_cfg["realizations"]:
            x, y = gmean_trajectory(ds, real, clim_t, w_lat)
            m, _ = np.polyfit(x, y, 1)
            slopes.append(m)
            xs_all.append(x)
            ys_all.append(y)
            print(f"    {real:16s} slope={m:.4f}  "
                  f"CO2_norm=[{x.min():.3f},{x.max():.3f}]  "
                  f"dT_norm=[{y.min():.3f},{y.max():.3f}]")
        slopes = np.array(slopes)
        x_pool = np.concatenate(xs_all)
        y_pool = np.concatenate(ys_all)
        m_pool, _ = np.polyfit(x_pool, y_pool, 1)
        return slopes, m_pool

    print(f"\n[ALL REALIZATIONS] hist:")
    slopes_h, pool_h = scan_all(hist_ds, hist_cfg)
    print(f"  mean={slopes_h.mean():.4f}  std={slopes_h.std():.4f}  "
          f"min={slopes_h.min():.4f}  max={slopes_h.max():.4f}  pooled_fit={pool_h:.4f}")
    print(f"  single-real slope {slope_h0:.4f} is "
          f"{(slope_h0 - slopes_h.mean()) / slopes_h.std() if slopes_h.std() > 0 else float('nan'):.2f} "
          f"std from the ensemble mean")

    print(f"\n[ALL REALIZATIONS] ssp370:")
    slopes_s, pool_s = scan_all(ssp370_ds, ssp370_cfg)
    print(f"  mean={slopes_s.mean():.4f}  std={slopes_s.std():.4f}  "
          f"min={slopes_s.min():.4f}  max={slopes_s.max():.4f}  pooled_fit={pool_s:.4f}")
    print(f"  single-real slope {slope_s0:.4f} is "
          f"{(slope_s0 - slopes_s.mean()) / slopes_s.std() if slopes_s.std() > 0 else float('nan'):.2f} "
          f"std from the ensemble mean")

    print(f"\n[VERDICT]")
    rel_err_s = 100.0 * (slope_s0 - slopes_s.mean()) / slopes_s.mean()
    rel_err_h = 100.0 * (slope_h0 - slopes_h.mean()) / slopes_h.mean()
    print(f"  ssp370 single-real vs ensemble-mean slope error: {rel_err_s:+.1f}%")
    print(f"  hist   single-real vs ensemble-mean slope error: {rel_err_h:+.1f}%")
    print("  If ssp370's error is large and positive, the SGAIN target under-states "
          "the true sensitivity by that much less than the model would need to warm "
          "-- i.e. the single-realization draw used for training happens to run cooler "
          "than the ensemble, so SGAIN pulls the model toward a target that is itself "
          "too cool, and the persistent warm bias could instead be the model correctly "
          "reverting toward the (warmer) ensemble-consistent trajectory once SGAIN's "
          "pull is weak (only 1/50 steps, scale 0.05).")


if __name__ == "__main__":
    main()
