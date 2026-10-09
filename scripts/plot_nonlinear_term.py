#!/usr/bin/env python3
"""Can the emulator reproduce N = ssp370 - GHG - AAER, and how much can we trust it?

N is the non-additivity of the single-forcing runs: the amount by which the
GHG-only and aerosol-only responses fail to sum to the all-forcing one. It is a
THREE-scenario residual, which is the quantity this emulator is weakest at --
its scenario TOTALS are accurate to 0-10% while DIFFERENCES between scenarios
degrade badly.

The figure makes one point that a bare N-vs-N comparison hides. The emulator's
N can match CESM2's closely while each of its three COMPONENTS is off by an
amount comparable to N itself, because the component errors partly CANCEL in
the residual. Nothing enforces that cancellation, so the honest uncertainty on
N is the component-error scale, not the apparent agreement.

  left    N(t), model vs CESM2, with a band of +-sum|component errors| -- what
          N's error would be if the component errors stopped cancelling
  middle  the three component anomalies, model vs CESM2
  right   the cancellation itself: each decade's component errors and the
          residual they leave in N

Convention: N = ssp370 - GHG - AAER on anomalies vs 1850-1900. N's SIGN is
convention-dependent, so the convention is printed on the figure.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_nonlinear_term.py --csv <decadal.csv>
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--csv", required=True, help="global_mean_anomaly_decadal.csv from an eval")
ap.add_argument("--label", default="mmlin sigma4")
ap.add_argument("--outdir", default="plots/nonlinear_term")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

rows = list(csv.DictReader(open(args.csv)))
D = {}
for r in rows:
    try:
        D[(r["experiment"], int(r["decade"]))] = (float(r["model_anom_degC"]),
                                                  float(r["cesm_anom_degC"]))
    except (ValueError, TypeError):
        continue

decs = sorted({d for (e, d) in D if e == "ssp370"}
              & {d for (e, d) in D if e == "ghg"}
              & {d for (e, d) in D if e == "aaer"})
if not decs:
    raise SystemExit("no decade has ssp370, ghg and aaer together")
print(f"[N] decades with all three single-forcing runs: {decs}")

ms = np.array([D[("ssp370", d)][0] for d in decs]); cs = np.array([D[("ssp370", d)][1] for d in decs])
mg = np.array([D[("ghg", d)][0] for d in decs]);    cg = np.array([D[("ghg", d)][1] for d in decs])
ma = np.array([D[("aaer", d)][0] for d in decs]);   ca = np.array([D[("aaer", d)][1] for d in decs])

n_m, n_c = ms - mg - ma, cs - cg - ca
e_s, e_g, e_a = ms - cs, mg - cg, ma - ca
n_err = e_s - e_g - e_a                      # the residual error actually left in N
# If the component errors did NOT cancel, N's error would be this large. This
# is the honest uncertainty: nothing makes the cancellation reliable.
band = np.abs(e_s) + np.abs(e_g) + np.abs(e_a)

x = np.arange(len(decs))
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)

# ── left: N, model vs CESM2, with the no-cancellation band ──────────────────
ax[0].fill_between(x, n_m - band, n_m + band, color="#B4451F", alpha=0.16,
                   label="model N $\\pm\\Sigma$|component errors|")
ax[0].plot(x, n_m, "o-", color="#B4451F", lw=2, label="emulator")
ax[0].plot(x, n_c, "s--", color="0.25", lw=2, label="CESM2")
for i in range(len(decs)):
    pct = 100 * (n_m[i] - n_c[i]) / n_c[i] if n_c[i] else np.nan
    ax[0].annotate(f"{pct:+.0f}%", (x[i], max(n_m[i], n_c[i])),
                   textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax[0].set_ylabel("N  [degC]")
ax[0].set_title("N = ssp370 $-$ GHG $-$ AAER\n(the single-forcing non-additivity)",
                fontsize=10)
ax[0].legend(fontsize=8, loc="upper left")

# ── middle: the three components ────────────────────────────────────────────
for lab, m, c, col in (("ssp370", ms, cs, "#B4451F"),
                       ("GHG", mg, cg, "#7E57C2"),
                       ("AAER", ma, ca, "#2F7D32")):
    ax[1].plot(x, m, "o-", color=col, lw=1.8, label=f"{lab} emulator")
    ax[1].plot(x, c, "s--", color=col, lw=1.2, alpha=0.6, label=f"{lab} CESM2")
ax[1].axhline(0, color="0.6", lw=0.6)
ax[1].set_ylabel("global-mean anomaly  [degC]")
ax[1].set_title("The three components\n(each accurate to a few percent)", fontsize=10)
ax[1].legend(fontsize=7, ncol=3, loc="upper left")

# ── right: the cancellation ─────────────────────────────────────────────────
wd = 0.26
ax[2].bar(x - wd, e_s, wd, label="ssp370 error", color="#B4451F")
ax[2].bar(x,      -e_g, wd, label="$-$GHG error", color="#7E57C2")
ax[2].bar(x + wd, -e_a, wd, label="$-$AAER error", color="#2F7D32")
ax[2].plot(x, n_err, "k*-", ms=11, lw=1.4, label="residual error in N")
ax[2].axhline(0, color="0.4", lw=0.8)
ax[2].set_ylabel("contribution to N's error  [degC]")
ax[2].set_title("Why N looks accurate: the component\nerrors CANCEL — nothing enforces that",
                fontsize=10)
ax[2].legend(fontsize=7.5)

for a in ax:
    a.set_xticks(x); a.set_xticklabels([f"{d}s" for d in decs])
    a.grid(alpha=0.3); a.set_xlabel("decade")

fig.suptitle(f"Nonlinear term N — {args.label}.  "
             f"Convention: N = ssp370 $-$ GHG $-$ AAER on anomalies vs 1850-1900 "
             f"(N's sign is convention-dependent).", fontsize=11)
out = os.path.join(args.outdir, "nonlinear_term_N")
for ext in (".png", ".pdf"):
    fig.savefig(out + ext, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}.png/.pdf")

print(f"\n{'decade':>8} {'N model':>9} {'N CESM2':>9} {'err':>8} {'err%':>7} "
      f"{'|band|':>8}  components: ssp370 / ghg / aaer errors")
print("-" * 96)
for i, d in enumerate(decs):
    pct = 100 * (n_m[i] - n_c[i]) / n_c[i] if n_c[i] else np.nan
    print(f"{d:>8} {n_m[i]:9.3f} {n_c[i]:9.3f} {n_err[i]:+8.3f} {pct:+6.0f}% "
          f"{band[i]:8.3f}   {e_s[i]:+.3f} / {e_g[i]:+.3f} / {e_a[i]:+.3f}")
print(f"\nN's apparent error is {np.abs(n_err).mean():.3f} degC on average, but the "
      f"component errors sum to {band.mean():.3f} degC.")
print("The agreement rests on cancellation, so quote N's uncertainty as the "
      "component scale, not the residual.")
