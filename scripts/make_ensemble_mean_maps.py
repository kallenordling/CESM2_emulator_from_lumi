#!/usr/bin/env python3
"""
Ensemble-mean MAPS over the same window the histogram figure pools.

paper_fig_histograms.py pools the final N years of every member into one
distribution. This draws the maps behind that number: the ensemble mean over the
same final-decade window, per experiment, emulator against held-out CESM2, with
their difference underneath.

Two forms of every figure, because they answer different questions:

  absolute  the mean state itself, degC and mm/day. The difference row then
            contains the emulator's mean-state offset (about -0.1 degC globally
            for this checkpoint), which is what the anomaly form removes.
  anomaly   each side minus ITS OWN 1850-1900 mean map, so a shared offset
            cancels and the difference row shows disagreement about the
            RESPONSE. Same convention as the timeseries and histogram figures:
            ssp370 has no pre-industrial of its own and inherits the historical
            baseline, on both sides.

REFERENCE = HELD-OUT MEMBERS ONLY, read from the training trees and truncated to
the emulator's ensemble size, exactly as the other paper figures do. The eval
NetCDF's own CESM arrays are not used: for aaer/ghg most of those members are
training data.

Maps are cached per variable — reading ~37 member trees over the mount takes
tens of minutes. Delete the .npz to force a re-read.

Usage
-----
    python scripts/make_ensemble_mean_maps.py                 # both vars, both forms
    python scripts/make_ensemble_mean_maps.py --var TREFHT
    python scripts/make_ensemble_mean_maps.py --n-years 20
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from scipy.stats import ttest_ind

# Cartopy is optional: it lives in the plotting env, not the base one, and the
# Natural Earth shapefiles it draws from are cached locally (no network at draw
# time). Without it the maps still render, on a plain lat/lon grid.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:                                   # pragma: no cover
    HAVE_CARTOPY = False
    print("[maps] cartopy not importable — falling back to plain axes")

PROJECTION = "Robinson"      # any ccrs class name taking no required arguments

# Where each figure lands in the paper. The anomaly pair is the main text; the
# absolute pair is the supplement, because an absolute map keeps the mean-state
# offset that the anomaly form removes by construction. Copies are written into
# plots/ under these names so the paper set and the working outputs cannot drift
# apart, and figures_overleaf/ takes the PDF of the same name.
PAPER_NAME = {
    ("TREFHT", "anomaly"): "fig05",     # reordered 2026-09-11, was fig09
    ("PRECT", "anomaly"): "fig06",      # reordered 2026-09-11, was fig10
    ("TREFHT", "absolute"): "figS03",
    ("PRECT", "absolute"): "figS04",
}

EVAL_DIR = "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25_absolute"
TREE_ROOT = "/home/nordling/mnt/lumi_sc/emulator_data/training_data"
DATA_CONFIG = "configs/config_data_ybias_BCprect.yaml"
BASELINE = (1850, 1900)

SCENARIOS = {"hist": ("Historical", "hist"),
             "ssp370": ("SSP3-7.0", "ssp370"),
             "aaer": ("Aerosol-only", "AAER"),
             "ghg": ("GHG-only", "GHG")}

META = {
    "TREFHT": dict(unit="degC", cmap="RdYlBu_r", dcmap="RdBu_r",
                   anom_cmap="RdBu_r", tree_scale=None),
    # LENS2 stores m/s, eval_aero writes mm/day: 1 m/s = 1000*86400 mm/day.
    "PRECT": dict(unit="mm/day", cmap="YlGnBu", dcmap="BrBG",
                  anom_cmap="BrBG", tree_scale=1000.0 * 86400.0),
}


def heldout_members(var):
    config = yaml.safe_load(open(DATA_CONFIG))
    trained = {e["scenario_name"]: set(e.get("realizations", []))
               for e in config["experiment_configs"]}
    out = {}
    for key, (_, subdir) in SCENARIOS.items():
        d = f"{TREE_ROOT}/{var}/{subdir}"
        on_disk = {n for n in os.listdir(d)
                   if n != "diagnostics" and os.path.isdir(f"{d}/{n}")}
        out[key] = sorted(on_disk - trained.get(key, set()))
    return out


def year_axis(da):
    return np.asarray(da["time" if "time" in da.dims else "year"].values).astype(int)


def window_mean(da, years, lo, hi):
    """Mean map over [lo, hi], or NaNs when the record does not reach it."""
    m = (years >= lo) & (years <= hi)
    if not m.any():
        return None
    dim = "time" if "time" in da.dims else "year"
    return np.asarray(da.isel({dim: np.where(m)[0]}).mean(dim).values, dtype=float)


def read_cesm(var, members, n_years):
    """Per experiment: (final-decade mean map, 1850-1900 mean map, window, n)."""
    scale = META[var]["tree_scale"]
    out = {}
    for key, (_, subdir) in SCENARIOS.items():
        finals, bases, wins = [], [], None
        for i, member in enumerate(members[key], 1):
            files = sorted(glob.glob(f"{TREE_ROOT}/{var}/{subdir}/{member}/*.nc"))
            print(f"  [{key}] {i}/{len(members[key])} {member}", flush=True)
            ds = xr.open_mfdataset(files, combine="by_coords", decode_times=False)
            da = ds[var]
            years = year_axis(da)
            hi = int(years.max())
            wins = (hi - n_years + 1, hi)
            f = window_mean(da, years, *wins)
            b = window_mean(da, years, *BASELINE)
            if scale is not None:
                f = f * scale
                b = b * scale if b is not None else None
            finals.append(f)
            bases.append(b)
            lat, lon = ds["lat"].values, ds["lon"].values
            ds.close()
        good_b = [b for b in bases if b is not None]
        out[key] = dict(final=np.mean(finals, axis=0),
                        base=np.mean(good_b, axis=0) if good_b else None,
                        # Per-member final-decade maps, kept for the significance
                        # test in draw(). The ensemble MEAN is what the figure
                        # shows; the spread is what decides whether a difference
                        # between the two means is distinguishable from noise.
                        final_members=np.stack(finals),
                        window=wins, n=len(finals), lat=lat, lon=lon)
    return out


def read_emulator(var, n_years, n_cap):
    out = {}
    for key in SCENARIOS:
        ds = xr.open_dataset(f"{EVAL_DIR}/{var}_{key}.nc")
        da = ds[f"{var}_model"].isel(member=slice(None, n_cap[key]))
        years = np.asarray(ds["year"].values).astype(int)
        hi = int(years.max())
        wins = (hi - n_years + 1, hi)
        sel = (years >= wins[0]) & (years <= wins[1])
        final = np.asarray(da.isel(year=np.where(sel)[0]).mean(("member", "year")).values,
                           dtype=float)
        bsel = (years >= BASELINE[0]) & (years <= BASELINE[1])
        base = (np.asarray(da.isel(year=np.where(bsel)[0]).mean(("member", "year")).values,
                           dtype=float) if bsel.any() else None)
        fm = np.asarray(da.isel(year=np.where(sel)[0]).mean("year")
                        .transpose("member", ...).values, dtype=float)
        out[key] = dict(final=final, base=base, window=wins,
                        final_members=fm,
                        n=int(da.sizes["member"]),
                        lat=ds["lat"].values, lon=ds["lon"].values)
        print(f"  [{key}] emulator {out[key]['n']} members, window {wins}")
        ds.close()
    return out


def to_celsius(side, var, name):
    """LENS2 stores kelvin; eval_aero writes Celsius. Put both in Celsius.

    The anomaly form hides this — a constant offset cancels in a difference of
    anomalies — so it has to be handled here or the absolute difference row is
    a flat -273. Detected, not assumed: a global mean near 287 is kelvin and one
    near 14 is Celsius, and nothing between 40 and 100 is either.
    """
    if var != "TREFHT":
        return side
    for key, d in side.items():
        typical = float(np.nanmean(d["final"]))
        if typical > 100:
            for field in ("final", "base", "final_members"):
                if d.get(field) is not None:
                    d[field] = d[field] - 273.15
            print(f"[units] {name:8s} {key:7s} K -> degC (mean was {typical:.1f})")
        elif typical > 40:
            raise SystemExit(f"[units] {name} {key}: mean {typical:.1f} is "
                             "neither kelvin (~287) nor Celsius (~14)")
    return side


def apply_baselines(side):
    """ssp370 has no pre-industrial of its own: it inherits the historical one."""
    if side["ssp370"]["base"] is None:
        side["ssp370"]["base"] = side["hist"]["base"]
    for key, d in side.items():
        if d["base"] is None:
            raise SystemExit(f"{key}: no 1850-1900 window on this side")
    return side


def to_pm180(field, lon):
    """Roll a 0-360 field onto -180..180, which is what PlateCarree expects."""
    lon = np.asarray(lon)
    if lon.max() <= 180.0:
        return field, lon
    shift = int((lon >= 180.0).sum())
    return np.roll(field, shift, axis=-1), np.roll(((lon + 180) % 360) - 180, shift)


def make_axes(fig, nrows, ncols):
    """A grid of map axes, projected when cartopy is available."""
    if not HAVE_CARTOPY:
        return fig.subplots(nrows, ncols, squeeze=False)
    proj = getattr(ccrs, PROJECTION)()
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(nrows, ncols, i * ncols + j + 1,
                                         projection=proj)
    return axes


# Panel letters, assigned row-major across the whole figure. A Robinson frame
# is elliptical, so the top-left corner of the axes box is off the globe and a
# letter there sits in white space rather than over data.
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _panel_letter(index):
    """a..z, then aa, ab, ... — a 7-row comparison has 28 panels, past 'z'."""
    n = len(PANEL_LETTERS)
    if index < n:
        return PANEL_LETTERS[index]
    return PANEL_LETTERS[index // n - 1] + PANEL_LETTERS[index % n]


def panel_label(ax, index):
    ax.text(0.0, 1.0, f"({_panel_letter(index)})", transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="top", ha="left")


def draw_map(ax, data, outline="0.15", **kw):
    """One panel: the field, then coastlines and country borders over it.

    `outline` is the line colour. Dark colourmaps need a light one, or the
    borders vanish into the field they are drawn over.
    """
    if not HAVE_CARTOPY:
        return ax.imshow(data, origin="lower", extent=[0, 360, -90, 90],
                         aspect="auto", **kw)
    im = ax.imshow(data, origin="lower", extent=[-180, 180, -90, 90],
                   transform=ccrs.PlateCarree(), **kw)
    ax.coastlines(resolution="110m", linewidth=0.35, color=outline)
    ax.add_feature(cfeature.BORDERS, linewidth=0.22, edgecolor=outline,
                   alpha=0.8)
    ax.set_global()
    return im


def draw(var, mode, emu, ref, outdir, stipple_on=True):
    unit = META[var]["unit"]
    lat, lon = emu["hist"]["lat"], emu["hist"]["lon"]

    fields, sig = {}, {}
    for key in SCENARIOS:
        e = emu[key]["final"] - (emu[key]["base"] if mode == "anomaly" else 0)
        c = ref[key]["final"] - (ref[key]["base"] if mode == "anomaly" else 0)
        fields[key] = tuple(to_pm180(f, lon)[0] for f in (e, c, e - c))
        # Significance is a property of the two ENSEMBLES, so it is the same
        # test in both modes -- a constant baseline offset shifts every member
        # of a side alike and cannot change a difference of means.
        if stipple_on and "final_members" in emu[key] and "final_members" in ref[key]:
            m, frac = significant_mask(emu[key], ref[key])
            sig[key] = (to_pm180(m, lon)[0], frac)

    # One scale for the top two rows so emulator and CESM2 are comparable, and
    # a separate symmetric scale for the difference row.
    top = np.concatenate([np.ravel(v[:2]) for v in fields.values()])
    diff = np.concatenate([np.ravel(v[2]) for v in fields.values()])
    if mode == "anomaly":
        tmax = np.nanpercentile(np.abs(top), 99)
        tlim = (-tmax, tmax)
    elif var == "TREFHT":
        tlim = (np.nanpercentile(top, 1), np.nanpercentile(top, 99))
    else:
        tlim = (0, np.nanpercentile(top, 99))
    dmax = np.nanpercentile(np.abs(diff), 99)
    tcmap = META[var]["anom_cmap"] if mode == "anomaly" else META[var]["cmap"]

    rows = ["Emulator", "CESM2 held-out", "Emulator - CESM2"]
    fig = plt.figure(figsize=(3.5 * len(SCENARIOS), 6.6), constrained_layout=True)
    axes = make_axes(fig, 3, len(SCENARIOS))
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None],
                              fields["hist"][0].shape)
    ncols = len(SCENARIOS)
    for j, (key, (label, _)) in enumerate(SCENARIOS.items()):
        for i in range(3):
            ax = axes[i][j]
            panel_label(ax, i * ncols + j)
            data = fields[key][i]
            if i < 2:
                im_top = draw_map(ax, data, cmap=tcmap, vmin=tlim[0], vmax=tlim[1])
            else:
                im_diff = draw_map(ax, data, cmap=META[var]["dcmap"],
                                   vmin=-dmax, vmax=dmax)
                if key in sig:
                    stipple(ax, sig[key][0], lat, to_pm180(data, lon)[1])
            if i == 0:
                w = emu[key]["window"]
                ax.set_title(f"{label}\n{w[0]}-{w[1]}  "
                             f"(n = {emu[key]['n']} / {ref[key]['n']})", fontsize=9)
            if j == 0:
                # A projected axis has no meaningful y-label, so the row name
                # goes beside it as text instead.
                ax.text(-0.04, 0.5, rows[i], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=10)
            # Below the panel, not inside it: a Robinson frame is elliptical,
            # so the corners of the axes box are off the globe and text there
            # gets clipped by the neighbouring panel.
            gm = np.average(data, weights=weights)
            note = f"{gm:.2f} {unit}"
            if i == 2:
                note = f"{gm:+.2f} {unit}"
                if key in sig:
                    note += f"   {100 * sig[key][1]:.0f}% sig."
            ax.text(0.5, -0.06, note,
                    transform=ax.transAxes, fontsize=8, ha="center", va="top")
    fig.colorbar(im_top, ax=list(axes[:2].ravel()), shrink=0.62,
                 label=f"{var} {'anomaly ' if mode == 'anomaly' else ''}({unit})")
    fig.colorbar(im_diff, ax=list(axes[2].ravel()), shrink=0.8,
                 label=f"difference ({unit})")
    proj_note = f" ({PROJECTION})" if HAVE_CARTOPY else ""
    sig_note = ("; stippling = difference significant at FDR q<0.05 (Welch across members)"
                if sig else "")
    fig.suptitle(f"{var} ensemble mean, final decade — "
                 f"{'anomaly vs 1850-1900' if mode == 'anomaly' else 'absolute'}"
                 f"{proj_note}{sig_note}", fontsize=12)
    # PNG to look at, PDF to \includegraphics — the paper set is vector.
    path = os.path.join(outdir, f"ensmean_map_{var}_{mode}.png")
    outs = [path, os.path.splitext(path)[0] + ".pdf"]
    paper = PAPER_NAME.get((var, mode))
    if paper:
        parent = os.path.dirname(outdir.rstrip("/")) or "plots"
        # Supplement figures share one folder; main-text ones get a folder each
        # (plots/fig05/fig05.pdf), matching every other paper figure, AND a copy
        # beside it. Both existed already and drifted apart -- write both.
        targets = ([os.path.join(parent, "supplement")] if paper.startswith("figS")
                   else [parent, os.path.join(parent, paper)])
        for target in targets:
            os.makedirs(target, exist_ok=True)
            outs += [os.path.join(target, f"{paper}.png"),
                     os.path.join(target, f"{paper}.pdf")]
    for out in outs:
        fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ", ".join(outs)


def significant_mask(emu_key, ref_key, q=0.05):
    """Cells where the emulator's ensemble mean differs from CESM2's beyond
    ensemble noise: Welch t-test across MEMBERS, then Benjamini-Hochberg.

    Welch rather than Student because the two sides have different member counts
    (25 emulator vs 6-11 held-out CESM2) and different spreads. BH because a
    naive p < 0.05 over ~55 000 cells paints ~2 700 false positives -- enough to
    look like a coherent region on a map. q is the false DISCOVERY rate: 5% of
    the stippled cells are expected to be spurious, not 5% of all cells.

    The baseline is the side's ensemble MEAN in both cases, so this tests the
    final-decade difference and treats the 1850-1900 offset as common-mode. That
    understates uncertainty slightly, but the baseline spread is far smaller than
    the final-decade spread, and using it per member is impossible for aaer/ghg,
    which borrow ssp370's baseline.

    Returns (mask, fraction of area stippled). NOT evidence of no bias where it
    is False -- see the TOST note in bias_equivalence_testing.
    """
    a = emu_key["final_members"] - emu_key["base"]
    b = ref_key["final_members"] - ref_key["base"]
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.zeros(a.shape[1:], bool), 0.0
    p = ttest_ind(a, b, axis=0, equal_var=False).pvalue
    flat = np.ravel(p)
    ok = np.isfinite(flat)
    mask = np.zeros_like(flat, bool)
    if ok.any():
        pv = np.sort(flat[ok])
        n = pv.size
        below = np.where(pv <= q * np.arange(1, n + 1) / n)[0]
        if below.size:
            mask[ok] = flat[ok] <= pv[below[-1]]
    return mask.reshape(p.shape), float(mask.mean())


def stipple(ax, mask, lat, lon, step=3):
    """Dots on the cells the test rejects, thinned so the field stays readable."""
    if not mask.any():
        return
    la, lo = np.asarray(lat), np.asarray(lon)
    sub = np.zeros_like(mask)
    sub[::step, ::step] = mask[::step, ::step]
    yy, xx = np.where(sub)
    kw = dict(transform=ccrs.PlateCarree()) if HAVE_CARTOPY else {}
    ax.scatter(lo[xx], la[yy], s=0.45, c="0.12", marker=".",
               linewidths=0, alpha=0.75, **kw)


def weighted_stats(e, c, lat):
    """Area-weighted pattern correlation, RMSE and bias between two maps.

    cos(lat) weights throughout: an unweighted number over this grid counts a
    polar cell as heavily as a tropical one, and the poles are exactly where
    the emulator's largest differences sit.
    """
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], e.shape).ravel()
    x, y = np.ravel(e), np.ravel(c)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((y - my) ** 2, weights=w))
    return dict(r=float(cov / (sx * sy)),
                rmse=float(np.sqrt(np.average((x - y) ** 2, weights=w))),
                bias=float(mx - my))


def write_table(var, stats, outdir):
    """A bare tabular for \\input, plain LaTeX — no booktabs assumed."""
    unit = META[var]["unit"].replace("degC", "$^{\\circ}$C")
    rows = []
    for key, (label, _) in SCENARIOS.items():
        a, n = stats[("absolute", key)], stats[("anomaly", key)]
        rows.append(f"{label} & {a['r']:.4f} & {a['rmse']:.3f} & {a['bias']:+.3f} & "
                    f"{n['r']:.4f} & {n['rmse']:.3f} & {n['bias']:+.3f} \\\\")
    tex = "\n".join([
        r"\begin{tabular}{|l|r|r|r|r|r|r|}",
        r"\hline",
        r"\textbf{Experiment} & \multicolumn{3}{c|}{\textbf{Absolute}} & "
        r"\multicolumn{3}{c|}{\textbf{Anomaly vs 1850--1900}} \\",
        r"\cline{2-7}",
        f" & $r$ & RMSE ({unit}) & Bias ({unit}) & $r$ & RMSE ({unit}) & Bias ({unit}) \\\\",
        r"\hline", *rows, r"\hline", r"\end{tabular}",
    ])
    path = os.path.join(outdir, f"ensmean_skill_{var}.tex")
    with open(path, "w") as fh:
        fh.write(f"% {var}: emulated vs held-out CESM2 ensemble-mean MAPS over the\n"
                 "% final decade of each experiment. r, RMSE and bias are all\n"
                 "% cos(lat)-weighted over the map, NOT over a global-mean series --\n"
                 "% r here is a spatial PATTERN correlation.\n"
                 "% Absolute keeps the mean-state offset; anomaly removes it by\n"
                 "% referencing each side to its own 1850-1900 map.\n"
                 "%\n"
                 "% DO NOT QUOTE THE ABSOLUTE r AS SKILL. An absolute map is\n"
                 "% dominated by the pole-to-equator gradient, which both sides\n"
                 "% share trivially, so r rounds to 1.0000 whatever the emulator\n"
                 "% got right. The anomaly r is the one that discriminates: it\n"
                 "% falls to 0.94 (aaer) and 0.75 (PRECT historical), where the\n"
                 "% response pattern is small against internal variability.\n"
                 f"% Generated by scripts/make_ensemble_mean_maps.py -- do not edit.\n")
        fh.write(tex + "\n")
    return path


def write_correlation_table(all_stats, species, outdir):
    """One table: pattern correlation AND RMSE, every experiment, both variables."""
    path = os.path.join(outdir, "ensmean_correlation.tex")
    ncol = 1 + 4 * len(species)
    # Built outside the f-string: a backslash cannot appear in an f-string
    # expression, and every LaTeX macro here is one.
    def unit_of(v):
        return META[v]["unit"].replace("degC", "$^{\\circ}$C")
    head = " & ".join(r"\multicolumn{4}{c|}{\textbf{" + v + "} (" + unit_of(v) + ")}"
                      for v in species)
    sub = " & ".join(["$r$ & RMSE & $r$ & RMSE"] * len(species))
    grp = " & ".join([r"\multicolumn{2}{c|}{Absolute} & \multicolumn{2}{c|}{Anomaly}"]
                     * len(species))
    rows = []
    for key, (label, _) in SCENARIOS.items():
        cells = []
        for v in species:
            for mode in ("absolute", "anomaly"):
                st = all_stats[(v, mode, key)]
                cells += [f"{st['r']:.4f}", f"{st['rmse']:.3f}"]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    tex = "\n".join([
        r"\begin{tabular}{|l|" + "r|r|r|r|" * len(species) + "}",
        r"\hline",
        r"\textbf{Experiment} & " + head + r" \\",
        r"\cline{2-" + str(ncol) + "}",
        " & " + grp + r" \\",
        r"\cline{2-" + str(ncol) + "}",
        " & " + sub + r" \\",
        r"\hline", *rows, r"\hline",
        r"\end{tabular}",
    ])
    with open(path, "w") as fh:
        fh.write("% Emulated vs held-out CESM2 ensemble-mean MAPS, final decade of\n"
                 "% each experiment. r is the spatial PATTERN correlation and RMSE\n"
                 "% the map error, both cos(lat)-weighted; RMSE is in the variable's\n"
                 "% own unit, degC and mm/day.\n"
                 "%\n"
                 "% The ABSOLUTE r is not a skill score. An absolute map is dominated\n"
                 "% by the pole-to-equator gradient that both sides share by\n"
                 "% construction, so it saturates near 1 regardless. Its RMSE column\n"
                 "% IS informative: it carries the mean-state offset that the anomaly\n"
                 "% form removes.\n"
                 "% Generated by scripts/make_ensemble_mean_maps.py -- do not edit.\n")
        fh.write(tex + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", nargs="+", default=["TREFHT", "PRECT"],
                    choices=["TREFHT", "PRECT"])
    ap.add_argument("--n-years", type=int, default=10,
                    help="final-decade window, matching paper_fig_histograms.py")
    ap.add_argument("--outdir", default="plots/ensmean_maps")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    all_stats = {}
    for var in args.var:
        # v2 keeps the per-member maps the significance test needs. The v1 cache
        # is left in place -- make_arm_comparison_maps.py still reads it.
        cache = os.path.join(args.outdir, f"cache_{var}_{args.n_years}y_v2.npz")
        members = heldout_members(var)
        n_cap = {k: len(v) for k, v in members.items()}
        if os.path.exists(cache):
            print(f"[{var}] reusing {cache}")
            z = np.load(cache, allow_pickle=True)
            emu, ref = z["emu"].item(), z["ref"].item()
        else:
            print(f"[{var}] emulator maps from {EVAL_DIR}")
            emu = read_emulator(var, args.n_years, n_cap)
            print(f"[{var}] CESM2 held-out maps from the trees (slow)")
            ref = read_cesm(var, members, args.n_years)
            np.savez(cache, emu=emu, ref=ref)
            print(f"[{var}] cached to {cache}")
        emu = apply_baselines(to_celsius(emu, var, "emulator"))
        ref = apply_baselines(to_celsius(ref, var, "CESM2"))
        stats = {}
        lat = emu["hist"]["lat"]
        for mode in ("absolute", "anomaly"):
            for key in SCENARIOS:
                e = emu[key]["final"] - (emu[key]["base"] if mode == "anomaly" else 0)
                c = ref[key]["final"] - (ref[key]["base"] if mode == "anomaly" else 0)
                stats[(mode, key)] = weighted_stats(e, c, lat)
                all_stats[(var, mode, key)] = stats[(mode, key)]
                st = stats[(mode, key)]
                print(f"[skill] {var:6s} {mode:8s} {key:7s} "
                      f"r={st['r']:.4f} rmse={st['rmse']:.3f} bias={st['bias']:+.3f} "
                      f"{META[var]['unit']}")
            print("wrote", draw(var, mode, emu, ref, args.outdir))
        print("wrote", write_table(var, stats, args.outdir))
    if len(args.var) > 1:
        print("wrote", write_correlation_table(all_stats, args.var, args.outdir))


if __name__ == "__main__":
    main()
