#!/usr/bin/env python3
"""Plot the conditioning maps as the MODEL actually sees them.

Replicates the exact training/eval cond pipeline — normalize (climate_dataset
.normalize, the shared EMISSIONS_PATHS min/max) → spatial smoothing
(smooth_cond_spatial, gaussian|median + per-channel sigma from config_data.yaml).
For each cond channel it shows, at a few years:
    row 0: normalized (pre-smoothing)
    row 1: what the MODEL receives (post-smoothing)
    row 2: removed = pre − post   (what the smoothing threw away)
so you can see how much regional structure survives the denoiser.

Run ON LUMI (needs data/climate_dataset + the /scratch cond files), e.g. in the
container:
    singularity exec <SIF> bash -c 'cd <repo> && python plot_cond_input.py --scenario aaer'
"""
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from data.climate_dataset import normalize, smooth_cond_spatial

DATA_CFG = "configs/config_data.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="aaer",
                    help="experiment whose train cond_file to use (hist/ssp370/aaer/ghg)")
    ap.add_argument("--cond-file", default=None, help="explicit cond .nc (overrides --scenario)")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="years to show (default: first, middle, last available)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dc = OmegaConf.load(DATA_CFG)
    cond_vars = OmegaConf.to_container(dc.cond_vars, resolve=True)
    _cs = dc.get("cond_smooth_sigma", None)
    sig = OmegaConf.to_container(_cs, resolve=True) if _cs is not None else [0] * len(cond_vars)
    if isinstance(sig, (int, float)):
        sig = [float(sig)] * len(cond_vars)
    method = dc.get("cond_smooth_method", "gaussian")

    # Resolve cond file
    cond_file = args.cond_file
    if cond_file is None:
        exps = list(dc.experiment_configs)
        match = next((e for e in exps if e["scenario_name"] == args.scenario), None)
        if match is None:
            raise SystemExit(f"scenario {args.scenario} not in config experiment_configs")
        cond_file = match["cond_file"]
    print(f"[COND] file={cond_file}")
    print(f"[COND] cond_vars={cond_vars}  sigma={sig}  method={method}")

    raw = xr.open_dataset(cond_file)
    time_dim = "time" if "time" in raw.dims else "year"
    if time_dim not in raw.dims and "year" in raw.dims:
        raw = raw.rename({"year": time_dim})
    lat = raw["lat"].values
    lon = raw["lon"].values
    nrm = raw[cond_vars].map(normalize)
    stacked = nrm.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"]).transpose(
        "var", time_dim, "lat", "lon")
    pre = stacked.values                          # (n_vars, T, H, W) normalized
    post = smooth_cond_spatial(pre.copy(), sig, method, cond_vars)   # what the model sees

    yrs = raw[time_dim].values
    yint = np.array([int(str(y)[:4]) for y in yrs]) if hasattr(yrs[0], "year") \
        else np.asarray(yrs, int)
    if args.years:
        cols = [int(np.argmin(np.abs(yint - y))) for y in args.years]
    else:
        cols = [0, len(yint) // 2, len(yint) - 1]
    show_years = [yint[c] for c in cols]

    n_ch = len(cond_vars)
    fig, axes = plt.subplots(3 * n_ch, len(cols),
                             figsize=(3.4 * len(cols), 2.4 * 3 * n_ch), squeeze=False)
    for ci, var in enumerate(cond_vars):
        vmax = float(np.nanmax(np.abs(pre[ci])))
        for j, c in enumerate(cols):
            rows = {0: ("normalized", pre[ci, c]),
                    1: ("model sees (smoothed)", post[ci, c]),
                    2: ("removed (pre−post)", pre[ci, c] - post[ci, c])}
            for r, (label, fld) in rows.items():
                ax = axes[ci * 3 + r, j]
                im = ax.pcolormesh(lon, lat, fld, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                   shading="auto")
                ax.set_title(f"{var} {show_years[j]} — {label}", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                if j == len(cols) - 1:
                    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Conditioning as the model sees it — {args.scenario} "
                 f"({method} σ={sig})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = args.out or f"cond_input_{args.scenario}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    # quick numeric summary of structure retained per channel
    for ci, var in enumerate(cond_vars):
        v0 = pre[ci].var(); vk = post[ci].var() / v0 if v0 > 0 else float("nan")
        pc = np.corrcoef(pre[ci].ravel(), post[ci].ravel())[0, 1]
        print(f"  {var}: var_kept={vk:.2f}  corr(pre,post)={pc:.3f}")


if __name__ == "__main__":
    main()
