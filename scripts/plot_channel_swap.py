#!/usr/bin/env python3
"""Where each conditioning channel's contribution to a scenario response lives.

Maps the Analysis B decomposition: one row per scenario, one column per
changed channel, then the INTERACTION term and the FULL response.

    R(X) = model(X) - model(ssp370)
         ~ sum_c [ model(ssp370 with channel c from X) - model(ssp370) ]  + interaction

The interaction column is the point of the figure. For RAMIP it is 59.6% of
the response -- larger than SUL or BC alone -- so the model's aerosol response
on a high-CO2 background is mostly a cross-term, not a sum of channels. For
ssp126/ssp245 it is 13-15% and the decomposition is effectively additive.

UNITS. The npz files were written before the denormalisation fix, so they hold
NORMALISED model space. DENORM_FN["TREFHT"] = x*21+4.5 and these are all
DIFFERENCES, so the offset cancels and a factor 21 converts them exactly. The
scaling is applied here and stated on the figure rather than silently.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_channel_swap.py
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

TREFHT_SCALE = 21.0        # DENORM_FN["TREFHT"] slope; offset cancels in a difference

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--indir", default=os.path.expanduser("~/mnt/lumi_sc/analysis/channel_swap"))
ap.add_argument("--outdir", default="plots/channel_swap")
ap.add_argument("--already-denormalised", action="store_true",
                help="set if the npz were written AFTER the denorm fix")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
SCALE = 1.0 if args.already_denormalised else TREFHT_SCALE

LABEL = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370-126aer": "RAMIP ssp370-126aer"}
ORDER = ["ssp370-126aer", "ssp126", "ssp245"]

runs = {}
for f in glob.glob(os.path.join(args.indir, "swap_*_vs_ssp370.npz")):
    key = os.path.basename(f)[len("swap_"):-len("_vs_ssp370.npz")]
    z = np.load(f, allow_pickle=True)
    parts = {k[len("part_"):]: z[k] * SCALE for k in z.files if k.startswith("part_")}
    runs[key] = dict(parts=parts, full=z["full"] * SCALE,
                     resid=z["residual"] * SCALE, lat=z["lat"], lon=z["lon"],
                     years=z["years"], members=int(z["members"]))
    print(f"[plot] {key}: channels {sorted(parts)} "
          f"{runs[key]['years'].min()}-{runs[key]['years'].max()}")
if not runs:
    raise SystemExit(f"no swap npz under {args.indir}")

keys = [k for k in ORDER if k in runs]
chans = ["CO2", "SUL", "BC"]
cols = chans + ["interaction", "FULL"]
lat = runs[keys[0]]["lat"]; lon = runs[keys[0]]["lon"]
lon180 = ((lon + 180) % 360) - 180
o = np.argsort(lon180)

w = np.cos(np.deg2rad(lat))[:, None]
def rms(a):
    return float(np.sqrt(np.average(a ** 2, weights=np.broadcast_to(w, a.shape))))

# One shared scale per ROW: contributions within a scenario must be comparable
# to each other. A global scale would flatten RAMIP, whose response is ~5x
# smaller than ssp126's.
subkw = {"projection": ccrs.Robinson()} if HAVE_CARTOPY else {}
fig, axes = plt.subplots(len(keys), len(cols),
                         figsize=(3.3 * len(cols), 2.5 * len(keys)),
                         constrained_layout=True, subplot_kw=subkw)
axes = np.atleast_2d(axes)

for i, k in enumerate(keys):
    r = runs[k]
    fields = {c: r["parts"].get(c) for c in chans}
    fields["interaction"] = r["resid"]
    fields["FULL"] = r["full"]
    vmax = float(np.nanpercentile(np.abs(r["full"]), 99.5)) or 1.0
    for j, c in enumerate(cols):
        ax = axes[i][j]
        f = fields[c]
        if f is None:                      # channel identical between scenarios
            ax.set_facecolor("0.94")
            ax.text(0.5, 0.5, "identical\nto ssp370", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="0.35")
            if HAVE_CARTOPY:
                ax.set_global()
        else:
            kw = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
            im = ax.pcolormesh(lon180[o], lat, f[:, o], cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax, shading="auto", **kw)
            if HAVE_CARTOPY:
                try:
                    ax.coastlines(linewidth=0.3, color="0.2")
                    ax.add_feature(cfeature.BORDERS, linewidth=0.15, edgecolor="0.5")
                    ax.set_global()
                except Exception as exc:                     # noqa: BLE001
                    print(f"[plot] coastline failed: {exc}")
            share = 100 * rms(f) / max(rms(r["full"]), 1e-30)
            ax.text(0.5, -0.06, f"rms {rms(f):.3f}   {share:.0f}% of full",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7)
        if i == 0:
            ax.set_title(c, fontsize=10,
                         fontweight="bold" if c in ("interaction", "FULL") else "normal")
        if j == 0:
            ax.text(-0.04, 0.5, f"{LABEL[k]}\n{r['years'].min()}-{r['years'].max()}",
                    transform=ax.transAxes, rotation=90, va="center", ha="right",
                    fontsize=8)
    fig.colorbar(im, ax=list(axes[i]), shrink=0.7, label="degC")

fig.suptitle("Scenario response decomposed by conditioning channel "
             "(model minus ssp370, 5 paired members)\n"
             "'interaction' is the non-additive residual — for RAMIP it is the "
             "LARGEST term; rows share a scale, columns do not",
             fontsize=11)
out = os.path.join(args.outdir, "channel_swap_maps")
for ext in (".png", ".pdf"):
    fig.savefig(out + ext, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}.png/.pdf")

print(f"\n{'scenario':>16} {'term':>12} {'rms degC':>10} {'% of full':>10}")
print("-" * 52)
for k in keys:
    r = runs[k]
    for c in chans:
        if r["parts"].get(c) is None:
            continue
        print(f"{k:>16} {c:>12} {rms(r['parts'][c]):10.4f} "
              f"{100*rms(r['parts'][c])/max(rms(r['full']),1e-30):9.0f}%")
    print(f"{k:>16} {'interaction':>12} {rms(r['resid']):10.4f} "
          f"{100*rms(r['resid'])/max(rms(r['full']),1e-30):9.0f}%")
    print(f"{k:>16} {'FULL':>12} {rms(r['full']):10.4f}")
