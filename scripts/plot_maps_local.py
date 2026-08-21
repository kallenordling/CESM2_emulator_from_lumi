#!/usr/bin/env python3
"""Redraw the eval maps LOCALLY, from NetCDFs an eval already wrote.

WHY THIS EXISTS. eval_aero.py draws its maps with cartopy, and cartopy is
broken in the project_462001112 venv:

    [PLOT-FAIL] plot_anomaly_maps: GEOSException: IllegalArgumentException:
                Points of LinearRing do not form a closed linestring

so every eval run from that project writes its NetCDFs and the pure-matplotlib
figures and then dies before the maps — 3 PNGs instead of 21. The DATA is fine;
only the drawing failed. This redraws from those files, on a machine whose
cartopy works, without regenerating anything.

Run it with the local plotting env (the base env has no cartopy):

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/plot_maps_local.py \
        --eval-dir /home/nordling/mnt/lumi_sc/eval_output/run_mseyb_BCprect/best_ep0430 \
        --var TREFHT --out-dir plots/

Reads over the sshfs mount, so no copying: lumi_sc is project_462001112 and
lumi_sc2 is project_462001328.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DEFAULT_YEARS = [1900, 1950, 2000, 2050, 2100]
STYLE = {
    "TREFHT": dict(units="°C", cmap="RdBu_r", vmax_anom=4.0, vmax_diff=2.0),
    "PRECT":  dict(units="mm/day", cmap="BrBG", vmax_anom=1.5, vmax_diff=1.0),
}


def load(eval_dir, var):
    out = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, f"{var}_*.nc"))):
        out[os.path.basename(f)[len(var) + 1:-3]] = xr.open_dataset(f)
    return out


def pick_years(d, requested, window):
    """Requested years that the file actually covers, given the averaging window."""
    yrs = d.year.values
    half = window // 2
    return [y for y in requested if (yrs >= y - half).any() and (yrs <= y + half).any()]


def mean_window(da, dim, year, window):
    half = window // 2
    sel = da.sel({dim: slice(year - half, year + half)})
    return sel.mean(dim) if sel.sizes[dim] else None


def make_axes(n_rows, n_cols, use_cartopy):
    if use_cartopy:
        import cartopy.crs as ccrs
        proj = ccrs.Robinson(central_longitude=180)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.6 * n_cols, 2.3 * n_rows),
                                 subplot_kw={"projection": proj}, squeeze=False)
        return fig, axes, ccrs.PlateCarree()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.3 * n_rows),
                             squeeze=False)
    return fig, axes, None


def draw(ax, field, lon, lat, cmap, vmax, transform):
    kw = dict(cmap=cmap, vmin=-vmax, vmax=vmax)
    if transform is not None:
        im = ax.pcolormesh(lon, lat, field, transform=transform, shading="auto", **kw)
        ax.coastlines(linewidth=0.4)
        ax.set_global()
    else:
        im = ax.imshow(field, origin="lower", extent=[0, 360, -90, 90],
                       aspect="auto", **kw)
    return im


def fig_for_scenario(scen, d, var, years, window, mode, out_path, use_cartopy):
    st = STYLE.get(var, STYLE["TREFHT"])
    suffix = "" if mode == "abs" else "_anom"
    mk, ck = f"{var}_model_mean{suffix}", f"{var}_cesm_mean{suffix}"
    if mk not in d:
        print(f"  [skip] {scen}: no {mk}")
        return
    has_cesm = ck in d
    lon, lat = d.lon.values, d.lat.values

    rows = ["Model"] + (["CESM2", "Model − CESM2"] if has_cesm else [])
    fig, axes, transform = make_axes(len(rows), len(years), use_cartopy)

    for c, y in enumerate(years):
        m = mean_window(d[mk], "year", y, window)
        panels = [(m, st["vmax_anom"] if mode != "abs" else None)]
        if has_cesm:
            cs = mean_window(d[ck], "cesm_year", y, window)
            panels.append((cs, st["vmax_anom"] if mode != "abs" else None))
            panels.append((m - cs if cs is not None else None, st["vmax_diff"]))

        for r, (field, vmax) in enumerate(panels):
            ax = axes[r][c]
            if field is None:
                ax.axis("off")
                continue
            if vmax is None:                      # absolute: data-driven range
                vmax = float(np.nanmax(np.abs(field.values)))
            im = draw(ax, field.values, lon, lat,
                      "RdBu_r" if r == 2 else st["cmap"], vmax, transform)
            if r == 0:
                ax.set_title(f"{y}", fontsize=10)
            if c == 0:
                ax.text(-0.08, 0.5, rows[r], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    label = "absolute" if mode == "abs" else "anomaly re 1850-1900"
    fig.suptitle(f"{scen} — {var} {label} [{st['units']}], "
                 f"{window}-year means", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WROTE] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True,
                    help="directory of <VAR>_<scenario>.nc (an sshfs mount is fine)")
    ap.add_argument("--var", default="TREFHT", choices=["TREFHT", "PRECT"])
    ap.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--window", type=int, default=10,
                    help="years averaged around each requested year (default 10)")
    ap.add_argument("--mode", choices=["anom", "abs"], default="anom",
                    help="anom = re 1850-1900 (the eval's own figures); "
                         "abs = the actual field the emulator outputs")
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--no-cartopy", action="store_true",
                    help="plain lat/lon, no coastlines — for when cartopy is broken")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    if not args.no_cartopy:
        try:
            import cartopy  # noqa: F401
        except ImportError:
            sys.exit("[FATAL] cartopy not importable. Use the plotting env:\n"
                     "  /home/nordling/miniconda3/envs/plotting/bin/python "
                     "scripts/plot_maps_local.py ...\n"
                     "or pass --no-cartopy for coastline-free maps.")

    runs = load(args.eval_dir, args.var)
    if not runs:
        sys.exit(f"[FATAL] no {args.var}_*.nc in {args.eval_dir}")
    if args.scenarios:
        runs = {k: v for k, v in runs.items() if k in args.scenarios}
    print(f"[LOAD] {len(runs)} scenario(s): {', '.join(runs)}")

    os.makedirs(args.out_dir, exist_ok=True)
    for scen, d in runs.items():
        years = pick_years(d, args.years, args.window)
        if not years:
            print(f"  [skip] {scen}: none of {args.years} within its record")
            continue
        out = os.path.join(args.out_dir, f"maps_{args.var}_{scen}_{args.mode}.png")
        fig_for_scenario(scen, d, args.var, years, args.window, args.mode,
                         out, not args.no_cartopy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
