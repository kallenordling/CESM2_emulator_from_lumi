#!/usr/bin/env python3
"""Maps of where the emulator puts the response to a regional aerosol change.

One row per region. Left: the full response, which is dominated by a UNIFORM
global-mean shift (halving a region's aerosols really does warm the planet).
Right: the same field with that global mean removed -- the pattern, which is
what "local emissions have a local effect" actually predicts and what the
locality metrics are computed on. The perturbed box is outlined on both.

Reading the left column alone is misleading: a globally uniform warming paints
the whole map one colour and looks like a strong response everywhere. The two
columns are deliberately on SEPARATE scales for that reason, with each panel's
own range printed, because forcing them onto one scale would make the pattern
invisible next to the global shift.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_regional_perturbation.py
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:                                            # noqa: BLE001
    HAVE_CARTOPY = False

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--indir",
                default=os.path.expanduser("~/mnt/lumi_sc/analysis/regional_perturbation"))
ap.add_argument("--outdir", default="plots/regional_perturbation")
ap.add_argument("--pattern", default="resp_*_x*_*.npz")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

files = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
if not files:
    raise SystemExit(f"no npz under {args.indir}")

runs = []
for f in files:
    z = np.load(f, allow_pickle=True)
    name = os.path.basename(f).replace("resp_", "").replace(".npz", "")
    runs.append(dict(name=name, resp=z["response"], pattern=z["pattern"],
                     gmean=float(z["gmean"]), lat=z["lat"], lon=z["lon"],
                     box=z["box"], inside=float(z["inside"]), r50=float(z["r50"])))
    print(f"[plot] {name}: gmean {runs[-1]['gmean']:+.4f} degC  "
          f"inside {runs[-1]['inside']:.1f}%  r50 {runs[-1]['r50']:.0f} km")

lat = runs[0]["lat"]
lon = runs[0]["lon"]
lon180 = ((lon + 180) % 360) - 180
order = np.argsort(lon180)


def shift(a):
    return a[:, order]


nrow = len(runs)
subkw = {"projection": ccrs.Robinson(central_longitude=0)} if HAVE_CARTOPY else {}
fig, axes = plt.subplots(nrow, 2, figsize=(11.5, 2.7 * nrow),
                         constrained_layout=True, subplot_kw=subkw)
axes = np.atleast_2d(axes)

for i, r in enumerate(runs):
    for j, (field, label) in enumerate(((r["resp"], "full response"),
                                        (r["pattern"], "pattern (global mean removed)"))):
        ax = axes[i][j]
        d = shift(field)
        v = float(np.nanpercentile(np.abs(d), 99.5))
        kw = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
        im = ax.pcolormesh(lon180[order], lat, d, cmap="RdBu_r",
                           vmin=-v, vmax=v, shading="auto", **kw)
        if HAVE_CARTOPY:
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="0.4")
                ax.coastlines(linewidth=0.35, color="0.15")
                ax.set_global()
            except Exception as exc:                          # noqa: BLE001
                print(f"[plot] border draw failed: {exc}")
        # the perturbed box
        la0, la1, lo0, lo1 = r["box"]
        bx = [lo0, lo1, lo1, lo0, lo0]
        by = [la0, la0, la1, la1, la0]
        ax.plot(bx, by, color="black", lw=1.4, zorder=5,
                **({"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}))
        fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02, label="degC")
        if i == 0:
            ax.set_title(label, fontsize=10)
        if j == 0:
            ax.text(-0.03, 0.5, r["name"].replace("_", " "), transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=10)
            ax.text(0.5, -0.07, f"global mean {r['gmean']:+.4f} degC",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8)
        else:
            ax.text(0.5, -0.07,
                    f"{r['inside']:.1f}% inside box   50% within {r['r50']:.0f} km",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8)

fig.suptitle("Response to halving one region's SUL+BC (2050, 25 paired members)\n"
             "box outlined; left is dominated by the uniform global-mean shift, "
             "right is the pattern locality is measured on", fontsize=11)
out = os.path.join(args.outdir, "regional_perturbation_maps")
for ext in (".png", ".pdf"):
    fig.savefig(out + ext, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}.png/.pdf")
