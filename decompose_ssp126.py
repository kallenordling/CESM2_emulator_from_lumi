#!/usr/bin/env python3
"""Model-side single-forcing decomposition of ssp126.

Runs the model on the ssp126 conditioning FOUR ways (same seed) to localise the
ssp126 over-warming:
    full     = f(CO2_ssp126, SUL_ssp126)   — the over-warming prediction
    ghg_only = f(CO2_ssp126, 0)            — SUL nulled (pre-industrial aerosol)
    sul_only = f(0, SUL_ssp126)            — CO2 nulled
    null     = f(0, 0)
Then, in global-mean ΔT relative to the null pass:
    dT_full, dT_ghg, dT_sul
    additive   = dT_ghg + dT_sul
    interaction = dT_full - additive   ( = full - ghg - sul + null )

Reading:
  * interaction ≈ 0  → ssp126 is composed additively (the #2 penalty reached it);
    any residual over-warming lives in a MARGINAL.
  * large dT_sul as ssp126 aerosols decline → the SUL (aerosol-removal) marginal
    is the culprit — which the interaction penalty cannot fix.
  * large interaction → the model still invents a CO2xaerosol interaction here.

Channels are nulled AFTER smoothing/PCA (NULL_COND = -1.0), matching the
trainer's co2only/sulonly passes. Run ON LUMI (model + GPU + data there):
    python decompose_ssp126.py                 # newest checkpoint
    python decompose_ssp126.py --checkpoint /path/to/ckpt.pt --fp32 --sample-steps 100
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import instantiate

import eval_aero as E   # reuse loaders / generator / constants


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="explicit ckpt; default = newest in --runs-dir")
    ap.add_argument("--runs-dir", default="/projappl/project_462001328/CESM2_emulator_from_lumi/runs")
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--out", default="ssp126_decomp.png")
    ap.add_argument("--cache", default=None,
                    help="npz cache of per-pass results (default auto-named from "
                         "checkpoint + sample-steps + precision). Existing passes "
                         "are reused; only missing ones are generated.")
    ap.add_argument("--force", action="store_true",
                    help="regenerate all passes, ignoring any cache")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    autocast_dt = None if args.fp32 else torch.bfloat16
    print(f"[DEVICE] {device} dtype={dtype} autocast={autocast_dt}")

    ckpt = args.checkpoint or E.find_latest_checkpoint(args.runs_dir)
    print(f"[CKPT] {ckpt}")
    model, pca_state = E.load_model(ckpt, E.CONFIG_PATH, device)
    model = model.to(dtype)
    pca_cond = pca_state.get("cond") if pca_state else None

    # Same cond preprocessing as eval (smoothing always; PCA only if ckpt carries it)
    data_cfg = OmegaConf.load("configs/config_data.yaml")
    _nc = data_cfg.get("n_components_cond", None)
    N_COMP = OmegaConf.to_container(_nc, resolve=True) if (pca_cond and _nc is not None) else None
    _cs = data_cfg.get("cond_smooth_sigma", None)
    SMOOTH = OmegaConf.to_container(_cs, resolve=True) if _cs is not None else None
    print(f"[COND] n_components_cond={N_COMP}  cond_smooth_sigma={SMOOTH}")

    ssp = next(e for e in E.EXPERIMENTS if e["name"] == "ssp126")
    cond, years, lat, lon = E.build_cond_tensor(
        ssp["cond_file"], E.COND_VARS, ssp["time_dim"], pca_cond, N_COMP, SMOOTH)
    print(f"[COND] ssp126 tensor {tuple(cond.shape)}  years {years.min()}-{years.max()}")

    scheduler = instantiate(OmegaConf.load(E.CONFIG_PATH).scheduler)

    NULL = E.NULL_COND
    co2only = cond.clone(); co2only[1] = NULL    # SUL nulled  → f(CO2,0)
    sulonly = cond.clone(); sulonly[0] = NULL    # CO2 nulled  → f(0,SUL)
    passes = {
        "full":     cond,
        "ghg_only": co2only,
        "sul_only": sulonly,
        "null":     torch.full_like(cond, NULL),
    }

    # ── Resumable cache: only generate passes that are missing ───────────────
    # Keyed by checkpoint + sample-steps + precision so a different setting
    # doesn't reuse stale results. Saved after every pass, so a timed-out or
    # re-submitted job continues where it left off.
    import os
    ckpt_tag = os.path.splitext(os.path.basename(ckpt))[0]
    prec = "fp32" if args.fp32 else "bf16"
    cache_path = args.cache or f"ssp126_decomp_{ckpt_tag}_s{args.sample_steps}_{prec}.npz"

    cache = {}
    if os.path.exists(cache_path) and not args.force:
        z = np.load(cache_path)
        cache = {k: z[k] for k in z.files}
        # invalidate if the cond grid changed (different year span)
        if "years" in cache and len(cache["years"]) != len(years):
            print(f"[CACHE] {cache_path} year span differs ({len(cache['years'])} vs "
                  f"{len(years)}) — ignoring")
            cache = {}
        else:
            have = [k for k in cache if k != "years"]
            print(f"[CACHE] {cache_path}: reusing passes {have}")
    cache["years"] = years

    gm = {}
    for name, c in passes.items():
        if name in cache and not args.force:
            gm[name] = cache[name]
            print(f"  [{name:8s}] cached — skip   (2100 gmean = {gm[name][-1]:.3f}°C)")
            continue
        gn = E.generate_timeseries(
            model, scheduler, c, device, dtype, args.sample_steps, args.batch_size,
            seed=0, autocast_dtype=autocast_dt)        # direct conditioning (w=1/1)
        celsius = gn * 21.0 + 4.5                       # denormalise → °C (T,H,W)
        gm[name] = E.area_weighted_gmean(celsius, lat)  # (T,)
        cache[name] = gm[name]
        np.savez(cache_path, **cache)                   # persist incrementally
        print(f"  [{name:8s}] generated — 2100 gmean = {gm[name][-1]:.3f}°C  (cached → {cache_path})")

    dT_full = gm["full"]     - gm["null"]
    dT_ghg  = gm["ghg_only"] - gm["null"]
    dT_sul  = gm["sul_only"] - gm["null"]
    dT_add  = dT_ghg + dT_sul
    interaction = dT_full - dT_add

    print("\n[DECOMP] global-mean ΔT (re null pass), 2100:")
    print(f"  full        = {dT_full[-1]:+.3f} °C")
    print(f"  ghg_only    = {dT_ghg[-1]:+.3f} °C")
    print(f"  sul_only    = {dT_sul[-1]:+.3f} °C   (aerosol-removal warming)")
    print(f"  additive    = {dT_add[-1]:+.3f} °C   (ghg + sul)")
    print(f"  interaction = {interaction[-1]:+.3f} °C   (full - additive)")
    frac = interaction[-1] / dT_full[-1] if dT_full[-1] != 0 else float("nan")
    print(f"  → interaction is {frac:+.0%} of full ΔT "
          f"({'MARGINAL-dominated' if abs(frac) < 0.25 else 'INTERACTION-dominated'})")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(years, dT_full, color="k",       lw=2.2, label="full  f(CO2,SUL)")
    ax.plot(years, dT_ghg,  color="#2ca02c", lw=1.8, label="GHG-only  f(CO2,0)")
    ax.plot(years, dT_sul,  color="#1f77b4", lw=1.8, label="SUL-only  f(0,SUL)")
    ax.plot(years, dT_add,  color="#d62728", lw=1.6, ls="--", label="additive  (GHG+SUL)")
    ax.plot(years, interaction, color="#9467bd", lw=1.4, ls=":", label="interaction  (full−additive)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("year")
    ax.set_ylabel("global-mean ΔT vs null pass (°C)")
    ax.set_title("ssp126 single-forcing decomposition (model-side)")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
