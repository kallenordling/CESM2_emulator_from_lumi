"""Plot emulator vs CESM2 global-mean anomaly from eval-generated NetCDFs.

Reads <output_dir>/<VAR>_<exp>.nc (written by eval_aero.py) for the user-chosen
experiments and overlays, per experiment:
  * emulator  = solid line, <VAR>_model_gmean_mean_anom (+ member min/max band)
  * CESM2     = dashed line, <VAR>_cesm_gmean_mean_anom  (+ member min/max band)
  * CMIP6 MMM = dotted line, the multimodel-mean <scn>_mmm.nc (from the cmip6 dir;
                --no-mmm to hide, --mmm-dir to point elsewhere)
A bottom panel shows the bias (emulator − CESM2) on common years.

Usage:
  python plot_from_nc.py <output_dir> <exp1> [exp2 ...] [--var TREFHT]
                         [--out FILE] [--no-spread] [--title TXT]
e.g.
  python plot_from_nc.py /mnt/lumi_sc2/eval_output/manual/ep0852_v2 ssp126 ssp245 ssp370
"""
import lumi_paths as L
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "hist": "#1f77b4", "ssp370": "#d62728", "ssp126": "#9467bd",
    "ssp245": "#17becf", "ghg": "#2ca02c", "aaer": "#ff7f0e",
}

# CMIP6 multimodel-mean global-mean tas anomaly files (in the cmip6 dir, NOT the
# eval output dir). Drawn as a dotted line per experiment.
MMM_FILES = {
    "hist": "historical_mmm.nc", "ssp126": "ssp126_mmm.nc",
    "ssp245": "ssp245_mmm.nc", "ssp370": "ssp370_mmm.nc",
}
# Default cmip6 locations to probe (local mount first, then LUMI scratch).
_MMM_DIR_CANDIDATES = [
    "/mnt/lumi_sc2/emulator_data/cmip6",
    f"{L.DATA}/cmip6",
]


def load_mmm(mmm_dir, scenario, baseline_cache):
    """CMIP6 multimodel-mean global-mean tas anomaly (re 1850-1900) → (years, anom)."""
    fn = MMM_FILES.get(scenario)
    if not fn or not mmm_dir:
        return None
    path = os.path.join(mmm_dir, fn)
    if not os.path.isfile(path):
        return None

    def _ann(p):
        ds = xr.open_dataset(p)
        t = ds["tas"]
        if "time" in t.dims:
            t = t.resample(time="YE").mean()
        yrs = _years(t)
        return yrs, np.asarray(t.values, dtype=float).reshape(len(yrs))

    try:
        if baseline_cache[0] is None:
            hp = os.path.join(mmm_dir, MMM_FILES["hist"])
            if os.path.isfile(hp):
                hy, hv = _ann(hp)
                m = (hy >= 1850) & (hy <= 1900)
                baseline_cache[0] = float(hv[m].mean()) if m.any() else 0.0
            else:
                baseline_cache[0] = 0.0
        y, v = _ann(path)
        return y, v - baseline_cache[0]
    except Exception as e:
        print(f"  [mmm] could not load {fn}: {e}")
        return None


def _years(da):
    d = "year" if "year" in da.dims else ("cesm_year" if "cesm_year" in da.dims else da.dims[0])
    v = da[d].values
    return (v.astype(int) if np.issubdtype(np.asarray(v).dtype, np.number)
            else np.array([int(str(x)[:4]) for x in v]))


def _members(ds, var_pat):
    """Stack all per-member series matching var_pat.format(i) → (N, T) or None."""
    mem, i = [], 1
    while var_pat.format(i) in ds:
        mem.append(ds[var_pat.format(i)].values)
        i += 1
    return np.stack(mem, axis=0) if mem else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("experiments", nargs="+", help="experiment names to plot")
    ap.add_argument("--var", default="TREFHT")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-spread", action="store_true", help="hide member min/max bands")
    ap.add_argument("--no-mmm", action="store_true", help="hide the CMIP6 multimodel-mean line")
    ap.add_argument("--mmm-dir", default=None,
                    help="dir with the *_mmm.nc files (default: probe the cmip6 dir)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    V = args.var

    mmm_dir = args.mmm_dir or next((d for d in _MMM_DIR_CANDIDATES if os.path.isdir(d)), None)
    mmm_base = [None]   # cached historical_mmm 1850-1900 mean

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    cyc = plt.cm.tab10(np.linspace(0, 1, 10))
    plotted = 0

    for k, exp in enumerate(args.experiments):
        path = os.path.join(args.output_dir, f"{V}_{exp}.nc")
        if not os.path.isfile(path):
            print(f"  [skip] not found: {path}")
            continue
        ds = xr.open_dataset(path)
        c = COLORS.get(exp, cyc[k % 10])

        # emulator (model)
        mg = ds[f"{V}_model_gmean_mean_anom"]
        my = _years(mg)
        ax.plot(my, mg.values, color=c, lw=2.0, label=f"{exp} emulator")
        if not args.no_spread:
            mm = _members(ds, f"{V}_model_gmean_m{{}}_anom")
            if mm is not None:
                ax.fill_between(my, mm.min(0), mm.max(0), color=c, alpha=0.15, lw=0)

        # CESM2
        cg_name = f"{V}_cesm_gmean_mean_anom"
        if cg_name in ds:
            cg = ds[cg_name]
            cy = _years(cg)
            ax.plot(cy, cg.values, color=c, lw=2.0, ls="--", alpha=0.85,
                    label=f"{exp} CESM2")
            if not args.no_spread:
                cm = _members(ds, f"{V}_cesm_gmean_m{{}}_anom")
                if cm is not None:
                    ax.fill_between(cy, cm.min(0), cm.max(0), facecolor="none",
                                    hatch="///", edgecolor=c, alpha=0.35, lw=0)
            # bias on common years
            common, im, ic = np.intersect1d(my, cy, return_indices=True)
            axb.plot(common, mg.values[im] - cg.values[ic], color=c, lw=1.5, label=exp)

        # CMIP6 multimodel mean (dotted)
        if not args.no_mmm:
            mmm = load_mmm(mmm_dir, exp, mmm_base)
            if mmm is not None:
                ax.plot(mmm[0], mmm[1], color=c, lw=1.3, ls=":", alpha=0.9,
                        label=f"{exp} CMIP6 MMM")
        plotted += 1
        ds.close()

    if not plotted:
        raise SystemExit("no experiments plotted (no matching NetCDFs found)")

    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_ylabel(f"{V} anomaly (°C)")
    ax.set_title(args.title or f"Emulator (solid) vs CESM2 (dashed) — {', '.join(args.experiments)}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    axb.axhline(0, color="k", lw=0.8)
    axb.set_ylabel("bias (emu − CESM2) °C"); axb.set_xlabel("year")
    axb.grid(alpha=0.3); axb.legend(fontsize=8)

    out = args.out or os.path.join(
        args.output_dir, f"timeseries_{'_'.join(args.experiments)}.png")
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
