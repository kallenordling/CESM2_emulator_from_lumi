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
    ap.add_argument("--experiment", default="hist")
    ap.add_argument("--vars", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--field", choices=["internal", "anomaly"], default="internal")
    ap.add_argument("--n-years", type=int, default=30,
                    help="use the last N years of the experiment")
    ap.add_argument("--space", choices=["physical", "normalized"],
                    default="physical",
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

    rows, spectra = [], {}
    for var in args.vars:
        p = os.path.join(args.eval_dir, f"{var}_{args.experiment}.nc")
        if not os.path.exists(p):
            print(f"[{var}] MISSING {p}")
            continue
        print(f"[{var}] reading {p} …", flush=True)
        E, ey, lat = load_side(p, var, "model", args.n_years)
        C, cy, _ = load_side(p, var, "cesm", args.n_years)
        if E is None or C is None:
            print(f"[{var}] no {'model' if E is None else 'cesm'} members — skipping")
            continue
        print(f"[{var}] emulator {E.shape[0]} members, CESM2 {C.shape[0]} members, "
              f"{args.n_years} yr, |lat|<={args.lat_max:g}")

        if args.space == "normalized":
            # Round-trip is exact for the emulator (the eval denormalised these
            # very arrays), and puts CESM2 in the same space, so the ratio is
            # computed where the loss actually acted.
            # scripts/ is not the repo root; put the root on the path so the
            # normalisation used in TRAINING is the one applied here, rather
            # than a copy that could drift from it.
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            from data.climate_dataset import NORM_FN
            if var not in NORM_FN:
                print(f"[{var}] no NORM_FN entry — cannot use --space normalized")
                continue
            E, C = NORM_FN[var](E), NORM_FN[var](C)

        Pe = (zonal_power(prepare(E, args.field), lat, args.lat_max)
              / dof_correction(args.field, E.shape[0]))
        Pc = (zonal_power(prepare(C, args.field), lat, args.lat_max)
              / dof_correction(args.field, C.shape[0]))
        print(f"[{var}] dof correction: emulator /{dof_correction(args.field, E.shape[0]):.3f}"
              f"  CESM2 /{dof_correction(args.field, C.shape[0]):.3f}")
        k = np.arange(1, len(Pe) + 1)
        ratio = Pe / Pc
        spectra[var] = (k, Pe, Pc, ratio)

        # Where does the emulator start losing power? Report the smallest
        # wavenumber beyond which the ratio stays below 0.8 — i.e. the scale at
        # which the deficit sets in and does not recover.
        below = ratio < 0.8
        k_break = None
        for i in range(len(k)):
            if below[i:].all():
                k_break = int(k[i]); break
        # total variance ratio, and the ratio restricted to the small scales
        small = k >= 20
        rows.append(dict(var=var, field=args.field, space=args.space,
                         experiment=args.experiment,
                         n_emu=E.shape[0], n_cesm=C.shape[0],
                         total_power_ratio=round(float(Pe.sum()/Pc.sum()), 4),
                         ratio_k1_5=round(float(Pe[:5].sum()/Pc[:5].sum()), 4),
                         ratio_k20plus=round(float(Pe[small].sum()/Pc[small].sum()), 4),
                         ratio_at_kmax=round(float(ratio[-1]), 4),
                         k_persistent_deficit=k_break))
        print(f"[{var}] power ratio  all k {rows[-1]['total_power_ratio']:.3f} | "
              f"k<=5 {rows[-1]['ratio_k1_5']:.3f} | k>=20 "
              f"{rows[-1]['ratio_k20plus']:.3f} | k_max {rows[-1]['ratio_at_kmax']:.3f}")

    if not spectra:
        print("nothing computed", file=sys.stderr)
        return 1

    t = pd.DataFrame(rows)
    print("\nZonal power, emulator / CESM2")
    print(t.to_string(index=False))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        long = pd.DataFrame([dict(var=v, wavenumber=int(kk), emulator=float(pe),
                                  cesm2=float(pc), ratio=float(r))
                             for v, (k, Pe, Pc, R) in spectra.items()
                             for kk, pe, pc, r in zip(k, Pe, Pc, R)])
        long.to_csv(args.csv, index=False)
        t.to_csv(args.csv.replace(".csv", "_summary.csv"), index=False)
        print(f"\nwrote {args.csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    n = len(spectra)
    fig, axes = plt.subplots(2, n, figsize=(4.6*n, 6.2), squeeze=False)
    for j, (var, (k, Pe, Pc, R)) in enumerate(spectra.items()):
        ax = axes[0][j]
        ax.loglog(k, Pc, color="#0072B2", lw=2, label="CESM2")
        ax.loglog(k, Pe, color="#D55E00", lw=2, label="Emulator")
        ax.set_title(f"({'ab'[j]})  {var}", loc="left", fontsize=10)
        ax.set_ylabel("zonal power")
        ax.legend(frameon=False)

        ax = axes[1][j]
        ax.semilogx(k, R, color="0.2", lw=2)
        ax.axhline(1.0, ls=":", lw=1, color="0.4")
        ax.axhline(0.8, ls="--", lw=0.8, color="0.6")
        ax.set_ylim(0, max(1.4, float(np.nanmax(R)) * 1.05))
        ax.set_xlabel("zonal wavenumber")
        ax.set_ylabel("emulator / CESM2")
        kb = t.loc[t["var"] == var, "k_persistent_deficit"].iloc[0]
        if kb is not None and not pd.isna(kb):
            ax.axvline(kb, color="#D55E00", ls="-.", lw=1)
            ax.text(kb, 0.05, f" deficit from k={int(kb)}", color="#D55E00",
                    fontsize=8, va="bottom")
    fig.suptitle(f"Zonal power spectra, {args.experiment}, "
                 f"{args.field} field, {args.space} space, "
                 f"last {args.n_years} yr, "
                 f"|lat| <= {args.lat_max:g}°\n"
                 f"ratio < 1 = emulator has LESS variance at that scale",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
