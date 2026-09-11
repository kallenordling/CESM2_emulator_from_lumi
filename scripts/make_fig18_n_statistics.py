#!/usr/bin/env python3
"""
================================================================================
 FIGURE 18 — THE FULL DESCRIPTION OF N: MAGNITUDE, VARIABILITY, SIGN, NOISE
================================================================================

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig16_n_statistics.py

WHY, BEYOND FIGURE 13
---------------------
Figure 11 shows mean(N) over one window. A mean can be small because the
nonlinearity is small or because it changes sign, and it says nothing about how
much of the map is forced signal rather than ensemble noise. This computes N for
EVERY year and derives the four quantities that separate those cases:

    mean(N)       the forced residual                (fig 11's quantity, full record)
    mean(|N|)     its typical magnitude              (large where mean(N) cancels)
    std(N)        its year-to-year variability
    P(N > 0)      the sign consistency               (0.5 = no consistent sign)

THE ONE THAT MATTERS MOST IS THE NOISE FLOOR
--------------------------------------------
N combines three ensemble MEANS, so at every grid point three sampling errors
add. That floor is computed here from the members themselves and reported as

    |mean(N)| / SE(N)

Where that ratio is below ~2, the residual at that point is not distinguishable
from the sampling noise of the three ensembles, however striking the colour is.
Precipitation is where this bites: its forced signal is small against its
internal variability, and a map that looks structured can be almost entirely
floor.

WHAT IT READS
-------------
The gridded NetCDFs, not the CSVs — per-YEAR fields are needed and the CSVs hold
global means only. Reduced to (year, lat, lon) per scenario immediately, so
nothing large is held: the member axis is collapsed into a mean and a variance
as each file is read.
"""

import os
import sys

# =============================================================================
#  SETTINGS
# =============================================================================

# Overridable so STEP 1 can run ON LUMI, where these files are local. Reducing
# them over the sshfs mount means pulling several GB through it and takes far
# longer than the job itself; on LUMI it is minutes. Run there with
#   N_EVAL_DIR=/scratch/project_462001112/eval_output/manual/ep0860_ens25_absolute \
#   N_REF_DIR=/scratch/project_462001112/emulator_data/cesm2_reference \
#   python scripts/make_fig16_n_statistics.py --reduce-only
# then scp the cache back and run again locally to plot.
EVAL_DIR = os.environ.get(
    "N_EVAL_DIR", "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25_absolute")
REFERENCE_DIR = os.environ.get(
    "N_REF_DIR", "/home/nordling/mnt/lumi_sc/emulator_data/cesm2_reference")

FIGURE_NAME = {"TREFHT": "fig18", "PRECT": "fig19"}
OUT = "plots/{name}/{name}.png"
TABLE = "plots/{name}/{name}_stats.tex"
NETCDF = "plots/{name}/{name}_fields.nc"

# Reducing eight files over the mount takes minutes; the reductions are small.
# DELETE after a new evaluation or a BASELINE change — keyed on neither.
CACHE = "plots/fig20_n_cache.npz"

BASELINE = (1850, 1900)
YEAR_MAX = 2050          # the single-forcing runs end here

MATCH_MEMBER_COUNTS = True

VARIABLES = {
    "TREFHT": ("Temperature", "$^{\\circ}$C", "$^{\\circ}$C", "RdBu_r"),
    "PRECT":  ("Precipitation", "mm day$^{-1}$", "mm\\,day$^{-1}$", "BrBG"),
}
INGREDIENTS = ("all", "ghg", "aaer")

REGIONS = [
    ("Arctic",       66.5,  90.0,   0.0, 360.0),
    ("N. Atlantic",  45.0,  65.0, 300.0, 350.0),
    ("N. America",   30.0,  60.0, 230.0, 300.0),
    ("Europe",       35.0,  65.0,   0.0,  40.0),
    ("East Asia",    20.0,  50.0, 100.0, 145.0),
    ("Sahel",        10.0,  20.0,   0.0,  40.0),
    ("Tropics",     -20.0,  20.0,   0.0, 360.0),
]

# =============================================================================

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (Agg must be set first)

REDUCE_ONLY = "--reduce-only" in sys.argv

# Cartopy is only needed for the maps, and STEP 1 runs on LUMI where the
# container has no cartopy. Import it lazily so --reduce-only works there.
if not REDUCE_ONLY:
    try:
        import cartopy.crs as ccrs
    except ImportError:
        sys.exit("[error] cartopy missing — use the plotting env "
                 "(/home/nordling/miniconda3/envs/plotting/bin/python)")

# =============================================================================
#  STEP 1 — per-year ensemble mean and variance, per scenario
# =============================================================================
# The member axis is collapsed as each file is read: an ensemble-mean map and a
# variance-of-the-mean map per year. Keeping members would be ~1 GB per file for
# no benefit — every quantity below is a function of those two.

def reduce_file(path, field_name, n_cap):
    dataset = xr.open_dataset(path)
    field = dataset[field_name]
    if n_cap is not None and field.sizes["member"] > n_cap:
        field = field.isel(member=slice(0, n_cap))
    n = field.sizes["member"]
    mean = field.mean("member").values.astype(np.float32)          # (year,lat,lon)
    # Variance OF THE MEAN — this is what adds when the three legs combine.
    var_mean = (field.var("member", ddof=1) / n).values.astype(np.float32)
    years = field["year"].values.astype(int)
    lat = field["lat"].values
    lon = field["lon"].values
    dataset.close()
    return mean, var_mean, years, lat, lon, n

store = {}
if os.path.exists(CACHE):
    loaded = np.load(CACHE, allow_pickle=False)
    for key in loaded.files:
        store[key] = loaded[key]
    print(f"[step 1] read {CACHE} — delete it to re-read the NetCDFs")
else:
    for variable in VARIABLES:
        for scenario in ("hist", "ssp370", "ghg", "aaer"):
            n_cesm = None
            for side, directory, name in (
                    ("cesm2", REFERENCE_DIR, f"{variable}_cesm"),
                    ("emulator", EVAL_DIR, f"{variable}_model")):
                cap = n_cesm if (side == "emulator" and MATCH_MEMBER_COUNTS) else None
                mean, var_mean, years, lat, lon, n = reduce_file(
                    f"{directory}/{variable}_{scenario}.nc", name, cap)
                if side == "cesm2":
                    n_cesm = n
                store[f"mean|{variable}|{scenario}|{side}"] = mean
                store[f"var|{variable}|{scenario}|{side}"] = var_mean
                store[f"years|{variable}|{scenario}|{side}"] = years
                print(f"[step 1] {variable:6s} {scenario:7s} {side:8s} "
                      f"{n} members, {len(years)} years", flush=True)
            store[f"lat|{variable}||"] = lat
            store[f"lon|{variable}||"] = lon
    np.savez_compressed(CACHE, **store)
    print(f"[step 1] wrote {CACHE}")

if REDUCE_ONLY:
    print("[step 1] --reduce-only: stopping before the figures. "
          "Copy the cache to plots/ and re-run without the flag.")
    sys.exit(0)

# =============================================================================
#  STEP 2 — N(x,t) per side, on the years all three legs cover
# =============================================================================
# ALL = hist spliced to ssp370 at 2015. Each leg is an anomaly against its OWN
# 1850-1900 ensemble mean, which is what makes the +Y0 control term unnecessary:
# N = (ALL-Y0) - (GHG-Y0) - (AAER-Y0) is exactly YGA - YG - YA + Y0.

results = {}
for variable in VARIABLES:
    lat = store[f"lat|{variable}||"]
    lon = store[f"lon|{variable}||"]
    weights2d = np.cos(np.deg2rad(lat))[:, None]

    for side in ("cesm2", "emulator"):
        series, variance = {}, {}
        for key in INGREDIENTS:
            if key == "all":
                hist_years = store[f"years|{variable}|hist|{side}"]
                ssp_years = store[f"years|{variable}|ssp370|{side}"]
                hist_mean = store[f"mean|{variable}|hist|{side}"]
                ssp_mean = store[f"mean|{variable}|ssp370|{side}"]
                hist_var = store[f"var|{variable}|hist|{side}"]
                ssp_var = store[f"var|{variable}|ssp370|{side}"]
                keep = hist_years < 2015
                years = np.concatenate([hist_years[keep], ssp_years])
                mean = np.concatenate([hist_mean[keep], ssp_mean])
                var = np.concatenate([hist_var[keep], ssp_var])
                # ssp370 has no pre-industrial of its own; the baseline is the
                # historical one, on both sides, which is what keeps them
                # comparable and what the splice requires.
                base_mask = (hist_years >= BASELINE[0]) & (hist_years <= BASELINE[1])
                baseline = hist_mean[base_mask].mean(axis=0)
            else:
                years = store[f"years|{variable}|{key}|{side}"]
                mean = store[f"mean|{variable}|{key}|{side}"]
                var = store[f"var|{variable}|{key}|{side}"]
                base_mask = (years >= BASELINE[0]) & (years <= BASELINE[1])
                baseline = mean[base_mask].mean(axis=0)
            series[key] = (years, mean - baseline, var)

        common = series["all"][0]
        for key in INGREDIENTS:
            common = np.intersect1d(common, series[key][0])
        common = common[common <= YEAR_MAX]

        stack, var_stack = {}, {}
        for key in INGREDIENTS:
            years, anomaly, var = series[key]
            index = np.searchsorted(years, common)
            stack[key] = anomaly[index]
            var_stack[key] = var[index]

        n_field = stack["all"] - stack["ghg"] - stack["aaer"]     # (T,lat,lon)
        # Three ensemble means combine, so three sampling variances add.
        se_field = np.sqrt(sum(var_stack[k] for k in INGREDIENTS))

        results[(variable, side)] = dict(
            years=common, N=n_field, se=se_field, lat=lat, lon=lon,
            weights=weights2d)
        print(f"[step 2] {variable:6s} {side:8s} N on "
              f"{common.min()}-{common.max()}, {n_field.shape}")

# =============================================================================
#  STEP 3 — the four statistics, plus the noise floor
# =============================================================================

for variable in VARIABLES:
    for side in ("cesm2", "emulator"):
        d = results[(variable, side)]
        n_field = d["N"]
        d["mean"] = n_field.mean(axis=0)
        d["absmean"] = np.abs(n_field).mean(axis=0)
        d["std"] = n_field.std(axis=0, ddof=1)
        d["p_positive"] = (n_field > 0).mean(axis=0)
        # SE of the TIME MEAN of N: the per-year sampling error averaged down,
        # treating years as independent, which they are not — so this is
        # optimistic and the ratio below is an upper bound on detectability.
        d["se_mean"] = np.sqrt((d["se"] ** 2).mean(axis=0) / len(d["years"]))
        d["ratio"] = np.abs(d["mean"]) / np.maximum(d["se_mean"], 1e-30)
        weights = np.broadcast_to(d["weights"], d["mean"].shape)
        d["area_resolved"] = 100 * float(np.average((d["ratio"] > 2).astype(float),
                                                    weights=weights))
        d["global_mean"] = float(np.average(d["mean"], weights=weights))
        d["global_absmean"] = float(np.average(d["absmean"], weights=weights))
        print(f"[step 3] {variable:6s} {side:8s} global mean(N) "
              f"{d['global_mean']:+.4f}, mean(|N|) {d['global_absmean']:.4f}, "
              f"{d['area_resolved']:.1f}% of area above 2x its noise floor")

# =============================================================================
#  STEP 4 — the figure: four panels per side
# =============================================================================

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "hatch.linewidth": 0.45})

for variable, (label, unit, unit_tex, cmap) in VARIABLES.items():
    figure_name = FIGURE_NAME[variable]
    fig = plt.figure(figsize=(13.4, 6.0))
    grid = fig.add_gridspec(2, 4, hspace=0.30, wspace=0.07)
    projection = ccrs.Robinson(central_longitude=0)

    for row, side in enumerate(("cesm2", "emulator")):
        d = results[(variable, side)]
        lat, lon = d["lat"], d["lon"]
        panels = [
            ("mean(N)", d["mean"], cmap, None),
            ("mean(|N|)", d["absmean"], "magma_r", "seq"),
            ("std(N) over time", d["std"], "viridis_r", "seq"),
            ("P(N > 0)", d["p_positive"], "PuOr_r", "prob"),
        ]
        for column, (title, field, colours, kind) in enumerate(panels):
            ax = fig.add_subplot(grid[row, column], projection=projection)
            if kind == "prob":
                kw = dict(vmin=0, vmax=1)
            elif kind == "seq":
                kw = dict(vmin=0, vmax=float(np.nanpercentile(field, 99)))
            else:
                v = float(np.nanpercentile(np.abs(field), 99))
                kw = dict(vmin=-v, vmax=v)
            image = ax.pcolormesh(lon, lat, field, cmap=colours, shading="auto",
                                  transform=ccrs.PlateCarree(), **kw)
            # On the mean panel, hatch where the residual is NOT resolved above
            # its own sampling floor. Hatching here means "do not read this".
            if kind is None:
                unresolved = (d["ratio"] <= 2).astype(float) + 1.0
                ax.contourf(lon, lat, unresolved, levels=[0.5, 1.5, 2.5],
                            colors="none", hatches=["", "...."],
                            transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.3, color="0.25")
            ax.set_global()
            if row == 0:
                ax.set_title(title, fontsize=10, pad=6)
            if column == 0:
                ax.text(-0.06, 0.5, "CESM2" if side == "cesm2" else "Emulator",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=10.5)
            fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05,
                         pad=0.04, aspect=18,
                         extend="both" if kind != "prob" else "neither")

    fig.suptitle(
        f"{label}: the nonlinearity N = ALL $-$ GHG $-$ AAER, "
        f"{results[(variable, 'cesm2')]['years'].min()}–"
        f"{results[(variable, 'cesm2')]['years'].max()}   ·   "
        f"hatching on mean(N) = NOT resolved above the three-ensemble "
        f"sampling floor", fontsize=11, y=1.0)

    out_path = OUT.format(name=figure_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for path in (out_path, os.path.splitext(out_path)[0] + ".pdf"):
        fig.savefig(path, bbox_inches="tight")
        print(f"[step 4] wrote {path}")
    plt.close(fig)

# =============================================================================
#  STEP 5 — NetCDF, so the fields are reusable without re-running
# =============================================================================

for variable in VARIABLES:
    figure_name = FIGURE_NAME[variable]
    d_c, d_e = results[(variable, "cesm2")], results[(variable, "emulator")]
    dataset = xr.Dataset(
        {f"{name}_{side}": (("lat", "lon"), results[(variable, s)][name])
         for name in ("mean", "absmean", "std", "p_positive", "se_mean", "ratio")
         for side, s in (("cesm2", "cesm2"), ("emulator", "emulator"))},
        coords={"lat": d_c["lat"], "lon": d_c["lon"]},
        attrs={"description": f"N = ALL - GHG - AAER statistics for {variable}",
               "years": f"{d_c['years'].min()}-{d_c['years'].max()}",
               "note": "hatched/unresolved where ratio <= 2"})
    dataset["N_cesm2"] = (("year", "lat", "lon"), d_c["N"])
    dataset["N_emulator"] = (("year", "lat", "lon"), d_e["N"])
    dataset = dataset.assign_coords(year=d_c["years"])
    path = NETCDF.format(name=figure_name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dataset.to_netcdf(path)
    print(f"[step 5] wrote {path}")

# =============================================================================
#  STEP 6 — regional table
# =============================================================================

for variable, (label, unit, unit_tex, _) in VARIABLES.items():
    figure_name = FIGURE_NAME[variable]
    d_c, d_e = results[(variable, "cesm2")], results[(variable, "emulator")]
    lat, lon = d_c["lat"], d_c["lon"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    weights2d = np.cos(np.deg2rad(lat))[:, None]

    rows_tex, printed = [], []
    for name, lat_min, lat_max, lon_min, lon_max in [("Global", -90, 90, 0, 360)] + REGIONS:
        inside = ((lat_grid >= lat_min) & (lat_grid <= lat_max)
                  & (lon_grid >= lon_min) & (lon_grid <= lon_max))
        w = np.where(inside, np.broadcast_to(weights2d, inside.shape), 0.0)
        cells = []
        for d in (d_c, d_e):
            cells += [float(np.average(d["mean"], weights=w)),
                      float(np.average(d["absmean"], weights=w)),
                      100 * float(np.average((d["ratio"] > 2).astype(float), weights=w))]
        rows_tex.append(f"{name} & " + " & ".join(
            f"{cells[i]:+.3f}" if i % 3 != 2 else f"{cells[i]:.0f}"
            for i in range(6)) + r" \\")
        printed.append((name, cells))
        print(f"[step 6] {variable:6s} {name:12s} CESM2 mean {cells[0]:+.3f} "
              f"|N| {cells[1]:.3f} resolved {cells[2]:.0f}%  |  "
              f"emu mean {cells[3]:+.3f} |N| {cells[4]:.3f} resolved {cells[5]:.0f}%")

    caption = (
        f"Nonlinearity statistics for {label.lower()} over "
        f"{d_c['years'].min()}--{d_c['years'].max()}, area-weighted by region. "
        f"``mean'' is the time mean of $N$ and ``$|N|$'' the time mean of its "
        f"absolute value: where $|N|$ greatly exceeds $|$mean$|$ the "
        f"nonlinearity is large but changes sign, which a mean map hides. "
        f"``Resolved'' is the percentage of the region's area where "
        f"$|$mean$(N)| $ exceeds twice its own sampling floor --- $N$ combines "
        f"three ensemble means, so three sampling errors add, and below that "
        f"threshold the residual cannot be distinguished from ensemble noise "
        f"however structured the map appears. Units {unit_tex}.")

    table_tex = "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + "}", r"\label{tab:" + figure_name + "}",
        r"\begin{tabular}{|l|r|r|r|r|r|r|}", r"\hline",
        r" & \multicolumn{3}{c|}{\textbf{CESM2}} & "
        r"\multicolumn{3}{c|}{\textbf{Emulator}} \\", r"\cline{2-7}",
        r"\textbf{Region} & mean & $|N|$ & resolved \% & "
        r"mean & $|N|$ & resolved \% \\", r"\hline",
        *rows_tex, r"\hline", r"\end{tabular}", r"\end{table}"])

    path = TABLE.format(name=figure_name)
    with open(path, "w") as handle:
        handle.write(f"% {label}: N statistics, full record.\n"
                     f"% Built by scripts/make_fig16_n_statistics.py — do not edit.\n")
        handle.write(table_tex + "\n")
    print(f"[step 6] wrote {path}")
