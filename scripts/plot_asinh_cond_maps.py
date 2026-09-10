#!/usr/bin/env python3
"""
Map what the asinh conditioning transform actually shows the model.

The emulator never sees emissions in Gt: it sees a normalised field in [-1, 1].
Which normaliser is in force therefore decides what spatial structure survives
into the network. This script draws that field, per channel, for chosen years,
under BOTH normalisers side by side:

    v1     (2v/hi - 1), clipped, hi a percentile of the WHOLE field
    asinh  asinh(v/s) / asinh(top/s), rescaled to [-1, 1]

Both are reimplemented here rather than imported, because data/climate_dataset.py
pulls in torch/accelerate, which a plotting box need not have. The formulas and
the percentile choices are copied from that module (see `normalize` and
`_get_emissions_minmax` there); if those change, change these.

The fit is done ONCE over the four training conditioning files (hist, ssp370,
aaer, ghg) exactly as training does, so a map here is the map the model gets.

Each panel is annotated with the share of POSITIVE cells that the transform
pins at +1 — the number the asinh arm exists to reduce.

Usage
-----
    python scripts/plot_asinh_cond_maps.py
    python scripts/plot_asinh_cond_maps.py --years 1850 2014 2050 2080 2100
    python scripts/plot_asinh_cond_maps.py --scenario ssp126 --species BC
    python scripts/plot_asinh_cond_maps.py --data-dir /scratch/project_462001328/emulator_data
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- copied from data/climate_dataset.py -------------------------------------
CLIP_PCTL = {"CO2": (1, 99), "SUL": (5, 95), "BC": (5, 95)}
ASINH_TOP_PCTL = 99.5     # positive-cell percentile that maps to +1
ASINH_SCALE_FRAC = 0.10   # s = this fraction of the positive-cell median
FIT_FILES = ["hist", "ssp370", "aaer", "ghg"]
SPECIES = ["CO2", "SUL", "BC"]
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR = os.path.expanduser("~/mnt/lumi_sc/emulator_data")

# The four emitting regions the v1 clip flattens hardest. A global pinned
# fraction hides them — most positive cells are near-zero ocean and land — so
# the regional table is reported next to it, never on its own.
REGIONS = {           # name: (lat0, lat1, lon0, lon1) with lon in [-180, 180]
    "E China": (20, 45, 100, 123),
    "India":   (8, 30, 70, 90),
    "E US":    (30, 45, -85, -70),
    "Europe":  (40, 60, -10, 30),
}


def cond_path(data_dir: str, tag: str) -> str:
    return os.path.join(data_dir, f"emissions_{tag}_only_timefixed_bc_co2fix.nc")


def fit_params(data_dir: str, species) -> dict:
    """(lo, hi) for v1 and (s, top) for asinh, over the training cond files."""
    pooled = {v: [] for v in species}
    for tag in FIT_FILES:
        with xr.open_dataset(cond_path(data_dir, tag)) as ds:
            for v in species:
                if v in ds.data_vars:
                    pooled[v].append(ds[v].values.ravel())
    params = {}
    for v in species:
        flat = np.concatenate(pooled[v])
        flat = flat[np.isfinite(flat)]
        pos = flat[flat > 0]
        plo, phi = CLIP_PCTL[v]
        params[v] = {
            "v1": (float(np.percentile(flat, plo)), float(np.percentile(flat, phi))),
            "asinh": (float(np.percentile(pos, 50)) * ASINH_SCALE_FRAC,
                      float(np.percentile(pos, ASINH_TOP_PCTL))),
        }
    return params


def apply_v1(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    if half == 0:
        return np.zeros_like(a)
    return np.clip((a - mid) / half, -1, 1)


def apply_asinh(a: np.ndarray, s: float, top: float) -> np.ndarray:
    denom = np.arcsinh(top / s)
    return np.clip(2.0 * np.arcsinh(a / s) / denom - 1.0, -1, 1)


def pinned_pct(raw: np.ndarray, norm: np.ndarray) -> float:
    """Share of emitting cells the transform flattens onto the +1 ceiling."""
    pos = raw > 0
    if not pos.any():
        return float("nan")
    return 100.0 * float((norm[pos] >= 1.0 - 1e-9).mean())


def region_mask(lat, lon, box):
    lat0, lat1, lon0, lon1 = box
    lon180 = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    m_lat = (lat >= lat0) & (lat <= lat1)
    m_lon = (lon180 >= lon0) & (lon180 <= lon1)
    return m_lat[:, None] & m_lon[None, :]


def load_year(data_dir: str, scenario: str, year: int, species) -> dict:
    tag = "hist" if year <= 2014 else scenario
    with xr.open_dataset(cond_path(data_dir, tag)) as ds:
        if year not in ds["year"].values:
            raise SystemExit(f"year {year} is not in {cond_path(data_dir, tag)}")
        sl = ds.sel(year=year)
        out = {v: np.asarray(sl[v].values, dtype=float) for v in species}
        out["_lat"] = ds["lat"].values
        out["_lon"] = ds["lon"].values
        out["_file"] = tag
    return out


def draw(species, years, frames, params, scenario, outdir):
    lat, lon = frames[years[0]]["_lat"], frames[years[0]]["_lon"]
    ext = [lon.min(), lon.max(), lat.min(), lat.max()]
    written = []
    for v in species:
        fig, axes = plt.subplots(2, len(years), figsize=(3.1 * len(years), 5.0),
                                 constrained_layout=True, squeeze=False)
        for j, yr in enumerate(years):
            raw = frames[yr][v]
            for i, (mode, fn, p) in enumerate((
                ("v1", apply_v1, params[v]["v1"]),
                ("asinh", apply_asinh, params[v]["asinh"]),
            )):
                z = fn(raw, *p)
                ax = axes[i][j]
                im = ax.imshow(z, origin="lower", extent=ext, vmin=-1, vmax=1,
                               cmap="magma", aspect="auto", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{yr}  ({frames[yr]['_file']})", fontsize=10)
                if j == 0:
                    ax.set_ylabel(mode, fontsize=11)
                ax.text(0.02, 0.04, f"pinned {pinned_pct(raw, z):.0f}%",
                        transform=ax.transAxes, fontsize=8, color="w")
        cb = fig.colorbar(im, ax=axes, shrink=0.85)
        cb.set_label("normalised conditioning value")
        s, top = params[v]["asinh"]
        lo, hi = params[v]["v1"]
        fig.suptitle(f"{v} conditioning channel as the model sees it — {scenario}\n"
                     f"v1 hi={hi:.3g}   asinh s={s:.3g}, top={top:.3g}",
                     fontsize=11)
        path = os.path.join(outdir, f"asinh_cond_map_{scenario}_{v}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def regional_table(species, years, frames, params, out_csv):
    """Share of each region's emitting cells pinned at +1, per transform."""
    lat, lon = frames[years[0]]["_lat"], frames[years[0]]["_lon"]
    masks = {name: region_mask(lat, lon, box) for name, box in REGIONS.items()}
    rows = []
    for v in species:
        for yr in years:
            raw = frames[yr][v]
            for mode, fn, p in (("v1", apply_v1, params[v]["v1"]),
                                ("asinh", apply_asinh, params[v]["asinh"])):
                z = fn(raw, *p)
                for name, m in masks.items():
                    rows.append((v, yr, mode, name,
                                 pinned_pct(raw[m], z[m])))
    with open(out_csv, "w") as fh:
        fh.write("species,year,transform,region,pinned_pct\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.1f}\n")
    hdr = "  ".join(f"{n:>8s}" for n in REGIONS)
    print(f"\npinned share of emitting cells (%)\n{'':22s}{hdr}")
    for v in species:
        for mode in ("v1", "asinh"):
            for yr in years:
                vals = [r[4] for r in rows
                        if r[0] == v and r[1] == yr and r[2] == mode]
                cells = "  ".join(f"{x:8.1f}" for x in vals)
                print(f"{v:4s} {mode:5s} {yr:4d}      {cells}")
    print("wrote", out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--scenario", default="ssp370")
    ap.add_argument("--years", type=int, nargs="+",
                    default=[1850, 2014, 2050, 2080, 2100])
    ap.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    ap.add_argument("--outdir", default="plots/asinh_cond")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    params = fit_params(args.data_dir, args.species)
    for v in args.species:
        lo, hi = params[v]["v1"]
        s, top = params[v]["asinh"]
        print(f"{v:4s} v1 (lo,hi)=({lo:.4g},{hi:.4g})   asinh (s,top)=({s:.4g},{top:.4g})")

    frames = {y: load_year(args.data_dir, args.scenario, y, args.species)
              for y in args.years}
    regional_table(args.species, args.years, frames, params,
                   os.path.join(args.outdir, f"pinned_by_region_{args.scenario}.csv"))
    for p in draw(args.species, args.years, frames, params, args.scenario, args.outdir):
        print("wrote", p)


if __name__ == "__main__":
    main()
