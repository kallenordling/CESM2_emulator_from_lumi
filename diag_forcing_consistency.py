"""Verify the single-forcing cond files match the combined hist+ssp370 forcing.

The model learns aerosol/GHG fingerprints from single-forcing experiments and
is expected to compose them back into the combined runs. That only works if the
conditioning the model is fed is actually CONSISTENT across experiments:

    aerosol (SUL):  aaer  cond  ==  hist+ssp370 cond   (GHGs held fixed in aaer)
    GHG     (CO2):  ghg   cond  ==  hist+ssp370 cond   (aerosols held fixed in ghg)

This script stitches the combined reference trajectory (hist for years ≤2014,
ssp370 for years ≥2015) per variable and differences it against the matching
single-forcing file, over their common years.

It compares the RAW cond fields. normalize() applies the SAME per-variable
affine map to every experiment, so raw-equal ⟺ model-input-equal (the only
downstream thing that can still differ is the per-scenario PCA basis — see
diag_cond_model_view.py). Pure xarray/numpy, no torch/omegaconf needed, so it
runs in a plain plotting env against the mounted files.

Outputs, per comparison:
  - <prefix>_<var>_maps.png   : single | reference | difference, across decades
  - <prefix>_<var>_timeseries.png : global-mean of each + per-year RMS difference
  - a printed summary (max/mean |diff|, worst years, held-fixed-var residual).

Usage:
    python diag_forcing_consistency.py
    EMU_DIR=/mnt/lumi_sc2/emulator_data python diag_forcing_consistency.py
    python diag_forcing_consistency.py --decades
"""
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMU_DIR = os.environ.get("EMU_DIR", "/mnt/lumi_sc2/emulator_data")
VAR_ALIASES = {"SUL": ["SUL", "sul", "SO2"], "CO2": ["CO2", "co2"]}


def resolve_var(ds: xr.Dataset, var: str):
    for alias in VAR_ALIASES.get(var, [var]):
        if alias in ds.data_vars:
            return alias
    raise KeyError(f"{var} (aliases {VAR_ALIASES.get(var)}) not in {list(ds.data_vars)}")


def time_dim(da: xr.DataArray) -> str:
    return next(d for d in da.dims if d not in ("lat", "lon"))


def years_of(da: xr.DataArray, tdim: str) -> np.ndarray:
    vals = da[tdim].values
    if hasattr(vals[0], "year") or not np.issubdtype(np.asarray(vals).dtype, np.number):
        return np.array([int(str(v)[:4]) for v in vals])
    return vals.astype(int)


def load_var(path: str, var: str) -> tuple[np.ndarray, np.ndarray, xr.DataArray]:
    """Return (years, data[T,H,W], DataArray) for `var` from a cond file."""
    ds = xr.open_dataset(path)
    name = resolve_var(ds, var)
    da = ds[name]
    tdim = time_dim(da)
    da = da.transpose(tdim, "lat", "lon")
    return years_of(da, tdim), da.values.astype(np.float64), da


def stitch_reference(hist_path, ssp_path, var) -> tuple[np.ndarray, np.ndarray, xr.DataArray]:
    """hist years ≤2014 + ssp370 years ≥2015, concatenated per gridpoint."""
    hy, hd, hda = load_var(hist_path, var)
    sy, sd, _   = load_var(ssp_path, var)
    hmask = hy <= 2014
    smask = sy >= 2015
    years = np.concatenate([hy[hmask], sy[smask]])
    data  = np.concatenate([hd[hmask], sd[smask]], axis=0)
    order = np.argsort(years)
    return years[order], data[order], hda  # hda only for lat/lon coords


def decade_or_spread(years: np.ndarray, decades: bool, n_cols: int) -> list[int]:
    if decades:
        lo = int(np.ceil(years.min() / 10.0) * 10)
        hi = int(np.floor(years.max() / 10.0) * 10)
        targets = list(range(lo, hi + 1, 10))
    else:
        targets = list(np.linspace(years.min(), years.max(), n_cols))
    return sorted({int(years[np.argmin(np.abs(years - t))]) for t in targets})


def area_gmean(field_thw: np.ndarray, lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat)); w = w / w.mean()
    return (field_thw * w[None, :, None]).mean(axis=(1, 2))


def compare(var, single_path, single_name, hist_path, ssp_path,
            held_var, prefix, decades, n_cols):
    print(f"\n========== {var}:  {single_name}  vs  hist+ssp370 ==========")
    sy, sd, sda = load_var(single_path, var)
    ry, rd, _   = stitch_reference(hist_path, ssp_path, var)
    lat = sda["lat"].values

    common = np.intersect1d(sy, ry)
    if common.size == 0:
        print(f"  !! no overlapping years ({single_name}: {sy.min()}-{sy.max()}, "
              f"ref: {ry.min()}-{ry.max()}) — skipping")
        return
    s_idx = {int(y): i for i, y in enumerate(sy)}
    r_idx = {int(y): i for i, y in enumerate(ry)}
    S = np.stack([sd[s_idx[int(y)]] for y in common])
    R = np.stack([rd[r_idx[int(y)]] for y in common])
    diff = S - R

    # ── summary ───────────────────────────────────────────────────────────────
    abs = np.abs(diff)
    scale = max(np.abs(R).max(), 1e-30)
    print(f"  overlap {common.min()}-{common.max()} ({common.size} yrs)  "
          f"ref|max|={np.abs(R).max():.3e}")
    print(f"  |diff|  max={abs.max():.3e}  mean={abs.mean():.3e}  "
          f"(relative to ref|max|: {abs.max()/scale*100:.2f}% / {abs.mean()/scale*100:.3f}%)")
    per_year_rms = np.sqrt((diff ** 2).mean(axis=(1, 2)))
    worst = np.argsort(per_year_rms)[::-1][:5]
    print("  worst years (RMS diff): " +
          ", ".join(f"{int(common[i])}={per_year_rms[i]:.2e}" for i in worst))
    if abs.max() / scale < 1e-4:
        print("  ✓ MATCH — single-forcing cond is identical to hist+ssp370.")
    else:
        print("  ✗ MISMATCH — the model sees DIFFERENT "
              f"{var} in {single_name} vs hist+ssp370 (see maps).")

    # held-fixed variable residual (aaer.CO2 / ghg.SUL should be ~0)
    try:
        _, hd_held, _ = load_var(single_path, held_var)
        print(f"  held-fixed {held_var} in {single_name}: "
              f"|max|={np.abs(hd_held).max():.3e}  mean={hd_held.mean():.3e}  "
              f"(expected ~0 for single-forcing)")
    except KeyError:
        print(f"  held-fixed {held_var}: not present in {single_name}")

    # ── maps figure ─────────────────────────────────────────────────────────--
    cols = decade_or_spread(common, decades, n_cols)
    cidx = [int(np.where(common == y)[0][0]) for y in cols]
    rows = [(f"{single_name}  {var}", S, False),
            (f"hist+ssp370  {var}", R, False),
            (f"diff ({single_name} − ref)", diff, True)]
    vmax_data = max(np.abs(S[cidx]).max(), np.abs(R[cidx]).max(), 1e-30)
    dmax = max(np.abs(diff[cidx]).max(), 1e-30)
    fig, axes = plt.subplots(3, len(cols), figsize=(3.6 * len(cols), 9), squeeze=False)
    for r, (label, arr, is_diff) in enumerate(rows):
        for c, (yr, ti) in enumerate(zip(cols, cidx)):
            ax = axes[r][c]
            if is_diff:
                im = ax.imshow(arr[ti], origin="lower", aspect="auto",
                               cmap="PuOr_r", vmin=-dmax, vmax=dmax)
            else:
                im = ax.imshow(arr[ti], origin="lower", aspect="auto",
                               cmap="viridis", vmin=0, vmax=vmax_data)
            if r == 0:
                ax.set_title(str(yr), fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"{var}: {single_name} vs hist+ssp370 — should be identical "
                 f"(diff row ≈ 0)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{prefix}_{var}_maps.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  [plot] {out}")

    # ── time series ───────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(common, area_gmean(S, lat), lw=2, label=f"{single_name}")
    ax1.plot(common, area_gmean(R, lat), lw=1.6, ls="--", label="hist+ssp370")
    ax1.axvline(2015, color="grey", ls=":", lw=1)
    ax1.set_ylabel(f"area-mean {var}"); ax1.set_title(f"{var}: global mean")
    ax1.grid(alpha=.3); ax1.legend()
    ax2.plot(common, per_year_rms, color="#d62728", lw=2)
    ax2.axvline(2015, color="grey", ls=":", lw=1)
    ax2.set_ylabel("per-year RMS diff"); ax2.set_xlabel("year")
    ax2.set_title("RMS gridpoint difference (single − ref) — spikes = inconsistency")
    ax2.grid(alpha=.3)
    plt.tight_layout()
    out = f"{prefix}_{var}_timeseries.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  [plot] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emu-dir", default=EMU_DIR)
    ap.add_argument("--hist",   default=None)
    ap.add_argument("--ssp370", default=None)
    ap.add_argument("--aaer",   default=None)
    ap.add_argument("--ghg",    default=None)
    ap.add_argument("--decades", action="store_true",
                    help="one map column per decade (default: 8 evenly-spaced)")
    ap.add_argument("--n-cols", type=int, default=8)
    ap.add_argument("--out-prefix", default="forcing_consistency")
    args = ap.parse_args()

    e = args.emu_dir
    hist   = args.hist   or f"{e}/emissions_hist_only_timefixed.nc"
    ssp370 = args.ssp370 or f"{e}/emissions_ssp370_only_timefixed.nc"
    aaer   = args.aaer   or f"{e}/emissions_aaer_only_timefixed.nc"
    ghg    = args.ghg    or f"{e}/emissions_ghg_only_timefixed.nc"
    for p in (hist, ssp370, aaer, ghg):
        if not os.path.exists(p):
            raise SystemExit(f"missing cond file: {p}")

    # aerosol consistency: aaer.SUL vs hist+ssp370.SUL  (aaer holds CO2 fixed)
    compare("SUL", aaer, "aaer", hist, ssp370,
            held_var="CO2", prefix=args.out_prefix,
            decades=args.decades, n_cols=args.n_cols)
    # GHG consistency: ghg.CO2 vs hist+ssp370.CO2  (ghg holds SUL fixed)
    compare("CO2", ghg, "ghg", hist, ssp370,
            held_var="SUL", prefix=args.out_prefix,
            decades=args.decades, n_cols=args.n_cols)


if __name__ == "__main__":
    main()
