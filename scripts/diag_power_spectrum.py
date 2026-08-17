#!/usr/bin/env python3
"""
Diagnostic: is the emulator's output too SMOOTH, and does it lose small scales
faster for precipitation than for temperature?

THE QUESTION
------------
Every spatial kernel in the model is (1,3,3) over three stride-2 downsamplings
(192x288 -> 96x144 -> 48x72 -> 24x36), which is a smoothing-biased operator
stack. Temperature anomaly fields are spectrally red and fit that bias well;
precipitation carries much more of its variance at small scales. If the
architecture (or the log1p+MSE loss, which pulls toward the conditional mean)
is suppressing fine structure, the signature is a POWER DEFICIT AT HIGH
WAVENUMBER — and a deficit that is worse for PRECT than for TREFHT.

If instead both variables track CESM2 out to the grid scale, the kernels are
exonerated and the precipitation gap has to be explained by detection power and
training coverage, not by architecture.

WHY ZONAL SPECTRA, NOT A 2-D FFT
--------------------------------
The grid is regular in lat/lon but NOT in physical distance: meridians converge
poleward, so a 2-D FFT mixes different physical wavelengths into the same
wavenumber bin and the result is dominated by grid geometry. A Fourier
transform along a LATITUDE CIRCLE is exact — the circle is periodic and evenly
sampled — so zonal wavenumber is physically meaningful at every latitude. Bands
are cos(lat)-weighted and the polar caps are excluded by default, where the
zonal wavelength shrinks toward zero.

WHICH FIELD
-----------
--field internal (default)
    Each member MINUS the ensemble mean of its own side, year by year. This is
    the generated INTERNAL VARIABILITY, and it is the direct test of "does the
    model produce realistic fine structure": the forced response cancels, so
    only the scale content of the noise remains.

--field anomaly
    Each member minus its own time mean, i.e. the full anomaly field including
    the forced pattern.

Both sides are treated identically, so the RATIO of power spectra is the
answer; absolute levels depend on the window and matter less.

Usage
-----
    python scripts/diag_power_spectrum.py --eval-dir <eval>/ep0490_ens25 \\
        --experiment hist --vars TREFHT PRECT
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr


def members(ds, prefix):
    names = sorted([v for v in ds.data_vars if re.fullmatch(rf"{prefix}_m\d+", v)],
                   key=lambda x: int(x.rsplit("_m", 1)[1]))
    return names


def load_side(path, var, side, n_years):
    """(member, year, lat, lon) for 'model' or 'cesm', last n_years."""
    ds = xr.open_dataset(path)
    names = members(ds, f"{var}_{side}")
    if not names:
        return None, None, None
    ydim = "year" if side == "model" else ("cesm_year" if "cesm_year" in ds.dims
                                           else "year")
    yrs = ds[ydim].values.astype(int)
    keep = yrs >= yrs.max() - n_years + 1
    M = np.stack([ds[n].isel({ydim: keep}).values for n in names])
    return M, yrs[keep], ds["lat"].values


def zonal_power(F, lat, lat_max):
    """(member, year, lat, lon) -> (wavenumber,) cos-lat-weighted mean power.

    rfft along longitude is exact on a latitude circle. Power is normalised by
    nlon^2 so it is comparable across latitudes, and k=0 (the zonal mean) is
    dropped because it carries no scale information.
    """
    band = np.abs(lat) <= lat_max
    F = F[:, :, band, :]
    w = np.cos(np.deg2rad(lat[band]))
    nlon = F.shape[-1]
    P = np.abs(np.fft.rfft(F, axis=-1)) ** 2 / nlon ** 2      # (m, y, lat, k)
    P = P.mean(axis=(0, 1))                                    # (lat, k)
    P = np.average(P, axis=0, weights=w)                       # (k,)
    return P[1:]                                               # drop k=0


def prepare(M, field):
    """Remove either the ensemble mean (internal) or the time mean (anomaly)."""
    if field == "internal":
        return M - M.mean(axis=0, keepdims=True)     # per-year ensemble mean
    return M - M.mean(axis=1, keepdims=True)         # per-member time mean


def dof_correction(field, n):
    """Undo the variance lost to estimating the mean that was subtracted.

    Deviations from a mean estimated from the SAME n samples have variance
    sigma^2 (1 - 1/n), not sigma^2. The two sides here have very different n
    (25 emulator vs 5-11 CESM2), so without this the ratio is biased upward by
    up to ~20% — enough to turn a real deficit into an apparent match.
    """
    if field == "internal":
        return 1.0 - 1.0 / n
    return 1.0          # time mean over many years; the 1/T bias is negligible


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--experiments", nargs="+", default=["hist"],
                    metavar="EXP",
                    help="one or more experiments, e.g. hist ssp370 aaer ghg. "
                         "Each is read once and both spaces are computed from "
                         "that read, since the files are GB-scale over a mount.")
    ap.add_argument("--vars", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--field", choices=["internal", "anomaly"], default="internal")
    ap.add_argument("--n-years", type=int, default=30,
                    help="use the last N years of the experiment")
    ap.add_argument("--space", choices=["physical", "normalized", "both"],
                    default="both",
                    help="'physical' = the field as stored (degC, mm/day). "
                         "'normalized' re-applies NORM_FN, i.e. the space the "
                         "model was actually TRAINED and scored in — for PRECT "
                         "that is (log1p(mm/day) - mean)/std. Comparing the two "
                         "separates a genuine failure to resolve small scales "
                         "(deficit in BOTH spaces) from one manufactured by the "
                         "convex expm1 back-transform (deficit in physical "
                         "space only). log1p is pointwise and exactly "
                         "invertible, so it cannot remove spatial variance by "
                         "itself; only its interaction with MSE can.")
    ap.add_argument("--lat-max", type=float, default=60.0,
                    help="exclude poleward of this, where zonal wavelength "
                         "collapses and the grid is badly anisotropic")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    tag = "" if args.space == "physical" else "_normspace"
    if args.out is None:
        args.out = f"plots/diag_power_spectrum{tag}.png"
    if args.csv is None:
        args.csv = f"plots/power_spectrum{tag}.csv"

    spaces = (["physical", "normalized"] if args.space == "both"
              else [args.space])
    # NORM_FN is the normalisation TRAINING used; import it rather than
    # re-implementing, so the two cannot drift apart.
    NORM_FN = None
    if "normalized" in spaces:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.climate_dataset import NORM_FN as _NF
        NORM_FN = _NF

    rows, spectra = [], {}
    for exp, var in [(e, v) for e in args.experiments for v in args.vars]:
        p = os.path.join(args.eval_dir, f"{var}_{exp}.nc")
        tagv = f"{exp}/{var}"
        if not os.path.exists(p):
            print(f"[{tagv}] MISSING {p}")
            continue
        print(f"[{tagv}] reading …", flush=True)
        E0, ey, lat = load_side(p, var, "model", args.n_years)
        C0, cy, _ = load_side(p, var, "cesm", args.n_years)
        if E0 is None or C0 is None:
            # ssp126/ssp245 PRECT have no CESM2 reference at all (model-only
            # panels); that is expected, not an error.
            print(f"[{tagv}] no {'model' if E0 is None else 'CESM2'} members "
                  f"— skipping")
            continue
        print(f"[{tagv}] emulator {E0.shape[0]} members, CESM2 {C0.shape[0]}, "
              f"{args.n_years} yr, |lat|<={args.lat_max:g}")

        for space in spaces:
          if space == "normalized":
            # Round-trip is exact for the emulator (the eval denormalised these
            # very arrays) and puts CESM2 in the same space, so the ratio is
            # computed where the loss actually acted.
            if var not in NORM_FN:
                print(f"[{tagv}] no NORM_FN entry — skipping normalized space")
                continue
            E, C = NORM_FN[var](E0), NORM_FN[var](C0)
          else:
            E, C = E0, C0

          # BOTH sides must be recomputed for every space. When this line sat
          # inside the else branch, the normalized pass reused Pe from the
          # physical pass while recomputing Pc, inflating the TREFHT ratio by
          # exactly 21^2 = 441 — the square of its (x-4.5)/21 normalisation.
          Pe = (zonal_power(prepare(E, args.field), lat, args.lat_max)
                / dof_correction(args.field, E.shape[0]))
          Pc = (zonal_power(prepare(C, args.field), lat, args.lat_max)
                / dof_correction(args.field, C.shape[0]))
          k = np.arange(1, len(Pe) + 1)
          ratio = Pe / Pc
          spectra[(exp, var, space)] = (k, Pe, Pc, ratio)

        # Where does the emulator start losing power? Report the smallest
        # wavenumber beyond which the ratio stays below 0.8 — i.e. the scale at
        # which the deficit sets in and does not recover.
          below = ratio < 0.8
          k_break = None
          for i in range(len(k)):
              if below[i:].all():
                  k_break = int(k[i]); break
          small = k >= 20
          rows.append(dict(experiment=exp, var=var, space=space,
                           field=args.field,
                           n_emu=E.shape[0], n_cesm=C.shape[0],
                           total_power_ratio=round(float(Pe.sum()/Pc.sum()), 4),
                           ratio_k1_5=round(float(Pe[:5].sum()/Pc[:5].sum()), 4),
                           ratio_k20plus=round(float(Pe[small].sum()/Pc[small].sum()), 4),
                           ratio_at_kmax=round(float(ratio[-1]), 4),
                           k_persistent_deficit=k_break))
          print(f"[{tagv}/{space:10s}] all k {rows[-1]['total_power_ratio']:.3f} | "
                f"k<=5 {rows[-1]['ratio_k1_5']:.3f} | k>=20 "
                f"{rows[-1]['ratio_k20plus']:.3f} | k_max "
                f"{rows[-1]['ratio_at_kmax']:.3f}")

    if not spectra:
        print("nothing computed", file=sys.stderr)
        return 1

    t = pd.DataFrame(rows).sort_values(["var", "space", "experiment"])
    print("\nZonal power, emulator / CESM2")
    print(t.to_string(index=False))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        long = pd.DataFrame([dict(experiment=e, var=v, space=sp,
                                  wavenumber=int(kk), emulator=float(pe),
                                  cesm2=float(pc), ratio=float(r))
                             for (e, v, sp), (k, Pe, Pc, R) in spectra.items()
                             for kk, pe, pc, r in zip(k, Pe, Pc, R)])
        long.to_csv(args.csv, index=False)
        t.to_csv(args.csv.replace(".csv", "_summary.csv"), index=False)
        print(f"\nwrote {args.csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    exps = [e for e in args.experiments
            if any(k[0] == e for k in spectra)]
    vars_ = [v for v in args.vars if any(k[1] == v for k in spectra)]
    nrow, ncol = len(vars_), len(exps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5*ncol, 3.0*nrow),
                             squeeze=False, sharey="row")
    # Ratio only: the ratio is the answer, and one panel per (var, experiment)
    # keeps the comparison ACROSS experiments readable, which is the point of
    # running more than one.
    style = {"physical": ("#D55E00", "-"), "normalized": ("#0072B2", "--")}
    for r, var in enumerate(vars_):
        for c, exp in enumerate(exps):
            ax = axes[r][c]
            drew = False
            for space in spaces:
                key = (exp, var, space)
                if key not in spectra:
                    continue
                k, Pe, Pc, R = spectra[key]
                col, ls = style[space]
                ax.semilogx(k, R, color=col, ls=ls, lw=1.8, label=space)
                drew = True
            if not drew:
                ax.set_visible(False); continue
            ax.axhline(1.0, ls=":", lw=1, color="0.4")
            ax.axhline(0.8, ls="--", lw=0.7, color="0.75")
            ax.set_ylim(0, 2.0)
            if r == 0:
                ax.set_title(exp, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{var}\nemulator / CESM2")
            if r == nrow - 1:
                ax.set_xlabel("zonal wavenumber")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Zonal power spectra, {args.field} field, last {args.n_years} yr, "
                 f"|lat| <= {args.lat_max:g}°\n"
                 f"ratio < 1 = emulator has LESS variance at that scale; "
                 f"'normalized' is the space the loss acted in",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
