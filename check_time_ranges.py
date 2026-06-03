"""Compare time-dim lengths/ranges between target climate files and cond files.

Reads experiment_configs from configs/config_data.yaml, opens each target
realization with xr.open_mfdataset and the matching cond file, then prints
their time-dim length and (min, max) year. Also reports the count under the
trainer's `selected_years` mask (every 5th hist year + every 2nd ssp year).
"""
import os
import glob
import argparse
from typing import Iterable

import numpy as np
import xarray as xr
import yaml


HIST_YEARS = list(range(1850, 2015, 5))
FUTURE_YEARS = list(range(2015, 2101, 2))
SELECTED = set(HIST_YEARS + FUTURE_YEARS)


def _years_from_coord(vals: np.ndarray) -> np.ndarray:
    if vals.size == 0:
        return vals.astype(int)
    v0 = vals[0]
    if hasattr(v0, "year"):
        return np.asarray([int(str(v)[:4]) for v in vals])
    if np.issubdtype(np.asarray(vals).dtype, np.integer):
        return vals.astype(int)
    if np.issubdtype(np.asarray(vals).dtype, np.floating):
        return vals.astype(int)
    return np.asarray([int(str(v)[:4]) for v in vals])


def _time_dim(ds: xr.Dataset) -> str:
    for d in ("time", "year"):
        if d in ds.dims:
            return d
    return next(d for d in ds.dims if d not in ("lat", "lon"))


def summarize(years: np.ndarray) -> str:
    if years.size == 0:
        return "EMPTY"
    sel = sorted(set(int(y) for y in years) & SELECTED)
    return (
        f"n={len(years):4d}  range={int(years.min())}-{int(years.max())}  "
        f"selected∩={len(sel):3d}"
    )


def _apply_rewrites(path: str, rewrites: list) -> str:
    for src, dst in rewrites:
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def open_target(data_dir: str, realization: str, rewrites: list) -> xr.Dataset:
    data_dir = _apply_rewrites(data_dir, rewrites)
    pattern = os.path.join(data_dir, realization, "*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return xr.open_mfdataset(files, combine="by_coords")


def open_cond(cond_file: str, rewrites: list) -> xr.Dataset:
    return xr.open_dataset(_apply_rewrites(cond_file, rewrites))


def check(experiments: Iterable[dict], realizations_per: int, rewrites: list) -> None:
    for exp in experiments:
        name = exp["scenario_name"]
        data_dir = exp["data_dir"]
        cond_file = exp["cond_file"]
        reals = list(exp["realizations"])[:realizations_per]
        print(f"\n=== {name} ===")
        print(f"  cond: {_apply_rewrites(cond_file, rewrites)}")
        try:
            cds = open_cond(cond_file, rewrites)
            ctd = _time_dim(cds)
            cyrs = _years_from_coord(cds[ctd].values)
            print(f"    [cond  dim={ctd:5s}] {summarize(cyrs)}")
        except Exception as e:
            print(f"    [cond] FAILED: {e}")
            cyrs = np.asarray([], dtype=int)

        for r in reals:
            try:
                tds = open_target(data_dir, r, rewrites)
                ttd = _time_dim(tds)
                tyrs = _years_from_coord(tds[ttd].values)
                tag = f"[target dim={ttd:5s}] {summarize(tyrs)}"
                if cyrs.size:
                    common = np.intersect1d(cyrs, tyrs)
                    sel_common = sorted(set(int(y) for y in common) & SELECTED)
                    diff = "OK" if len(sel_common) == len(set(tyrs) & SELECTED) == len(set(cyrs) & SELECTED) else "MISMATCH"
                    tag += f"  overlap_selected={len(sel_common):3d}  {diff}"
                print(f"    {r:<18s} {tag}")
            except Exception as e:
                print(f"    {r:<18s} FAILED: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_data.yaml")
    p.add_argument("--realizations-per", type=int, default=1,
                   help="how many realizations to inspect per scenario")
    p.add_argument("--include-val", action="store_true",
                   help="also check val_experiment_configs")
    p.add_argument("--path-prefix", action="append", default=[],
                   metavar="SRC=DST",
                   help="rewrite path prefix; repeatable. "
                        "e.g. --path-prefix /scratch/project_462001328=/mnt/lumi_sc2")
    args = p.parse_args()

    rewrites = []
    for spec in args.path_prefix:
        if "=" not in spec:
            raise SystemExit(f"--path-prefix expects SRC=DST, got: {spec}")
        src, dst = spec.split("=", 1)
        rewrites.append((src, dst))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("# train experiments")
    check(cfg["experiment_configs"], args.realizations_per, rewrites)
    if args.include_val and "val_experiment_configs" in cfg:
        print("\n# val experiments")
        check(cfg["val_experiment_configs"], args.realizations_per, rewrites)


if __name__ == "__main__":
    main()
