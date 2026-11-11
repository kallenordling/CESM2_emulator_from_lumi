#!/usr/bin/env python3
"""Turn a training checkpoint into the slim bundle the web app serves.

Run from the repo root (needs the repo's data/ and models/ packages):

    python webapp/export_bundle.py \
        --ckpt   ~/mnt/lumi_sc3/runs_mmlin/ens25/run_mmlin_co2fix_960.pt \
        --eval-dir ~/mnt/lumi_sc3/eval_output/manual/ens25_mmlin_ep960 \
        --cond-dir ~/mnt/lumi_sc3/emulator_data \
        --out webapp/bundle

Writes to --out:
  meta.json       model/scheduler kwargs, cond vars, sigma, ANCHORS (from the
                  checkpoint's COND_PROCESSED_NORM), target denorm constants
  weights.pt      the EMA weights as a plain model state_dict (what eval uses)
  cond_base.npz   RAW (unsmoothed) CO2/SUL/BC maps per scenario at the offered
                  years; the app edits these, then smooths + normalises
  baseline.npz    emulator 1850-1900 climatology (TREFHT degC, PRECT mm/day),
                  25-member mean of the eval, for anomalies
  reference.npz   25-member eval ensemble-mean maps at the offered years

Guards: refuses a checkpoint whose conditioning pipeline is not
smooth -> min/max (no PCA), whose sigma disagrees with the data config, or that
lacks persisted anchors; and checks the app's normalise() against
data.climate_dataset's own smooth_cond_spatial + normalize_tensor_cond.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "webapp", "space"))

SCENARIOS = {          # name -> cond file
    "hist":   "emissions_hist_only_timefixed_bc_co2fix.nc",
    "ssp126": "emissions_ssp126_only_timefixed_bc_co2fix.nc",
    "ssp245": "emissions_ssp245_only_timefixed_bc_co2fix.nc",
    "ssp370": "emissions_ssp370_only_timefixed_bc_co2fix.nc",
}
BASELINE = (1850, 1900)


def years_of(ds, dim):
    v = ds[dim].values
    if hasattr(v[0], "year"):
        return np.array([int(x.year) for x in v])
    return np.asarray(v, dtype=int)


def offered_years(all_years, step):
    lo, hi = int(all_years.min()), int(all_years.max())
    ys = [y for y in all_years if (y % step == 0) or y in (lo, hi)]
    return np.array(sorted(set(int(y) for y in ys)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval-dir", required=True,
                   help="ens25 eval of the SAME checkpoint (TREFHT_*.nc, PRECT_*.nc)")
    p.add_argument("--cond-dir", required=True)
    p.add_argument("--model-config", default=os.path.join(REPO, "configs/config_aero.yaml"))
    p.add_argument("--data-config",
                   default=os.path.join(REPO, "configs/config_data_ybias_BCprect_nopca.yaml"))
    p.add_argument("--out", default=os.path.join(REPO, "webapp/bundle"))
    p.add_argument("--year-step", type=int, default=5)
    p.add_argument("--half", action="store_true", help="store weights in fp16")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from data.climate_dataset import (PRECT_LOG_MEAN, PRECT_LOG_STD,
                                      normalize_tensor_cond,
                                      set_processed_minmax_override,
                                      smooth_cond_spatial)
    from models.video_net import UNetModel3D

    # ── checkpoint + guards ─────────────────────────────────────────────────
    print(f"[ckpt] loading {a.ckpt}")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if (ck.get("COND_TRANSFORM") or "minmax") != "minmax" or \
       (ck.get("COND_ORDER") or "normalize_last") != "normalize_last":
        sys.exit(f"[guard] cond pipeline {ck.get('COND_TRANSFORM')}/"
                 f"{ck.get('COND_ORDER')} is not minmax/normalize_last")
    pca = ck.get("PCA") or {}
    bases = [pca.get("cond"), pca.get("target")]
    for e in (pca.get("per_scenario") or {}).values():
        bases += [e.get("cond"), e.get("target")]
    if any(b is not None for b in bases):
        sys.exit("[guard] checkpoint has fitted PCA bases; the app has no PCA path")
    anchors = ck.get("COND_PROCESSED_NORM")
    if not anchors:
        sys.exit("[guard] checkpoint has no COND_PROCESSED_NORM anchors")

    mcfg = OmegaConf.to_container(OmegaConf.load(a.model_config), resolve=False)
    dcfg = OmegaConf.to_container(OmegaConf.load(a.data_config), resolve=False)
    model_kw = {k: v for k, v in mcfg["model"].items() if k != "_target_"}
    sched_kw = {k: v for k, v in mcfg["scheduler"].items() if k != "_target_"}
    cond_vars = list(dcfg["cond_vars"])
    target_vars = list(dcfg["target_vars"])
    sigma_cfg = [float(s) for s in dcfg["cond_smooth_sigma"]]
    sigma_ck = ck.get("COND_SIGMA")
    if sigma_ck is not None and [float(s) for s in sigma_ck] != sigma_cfg:
        sys.exit(f"[guard] checkpoint sigma {sigma_ck} != data config {sigma_cfg}")
    if dcfg.get("cond_smooth_method", "gaussian") != "gaussian":
        sys.exit("[guard] only gaussian smoothing is implemented in the app")
    print(f"[ckpt] step={ck.get('Global Step')} sigma={sigma_cfg} anchors={anchors}")

    model = UNetModel3D(**model_kw)
    missing, unexpected = model.load_state_dict(ck["EMA"], strict=False)
    if missing:
        sys.exit(f"[guard] {len(missing)} missing keys, e.g. {missing[:3]}")
    print(f"[ckpt] EMA loaded; {len(unexpected)} unexpected keys dropped")
    sd = model.state_dict()
    if a.half:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(sd, os.path.join(a.out, "weights.pt"))
    n_par = sum(v.numel() for v in model.parameters())
    del ck, model

    # ── conditioning ────────────────────────────────────────────────────────
    cond_arrays, lat, lon = {}, None, None
    set_processed_minmax_override(anchors)
    from emulator import Emulator  # noqa: F401  (import check only)
    for scen, fname in SCENARIOS.items():
        path = os.path.join(os.path.expanduser(a.cond_dir), fname)
        ds = xr.open_dataset(path)
        dim = "time" if "time" in ds.dims else "year"
        ys_all = years_of(ds, dim)
        ys = offered_years(ys_all, a.year_step)
        idx = [int(np.where(ys_all == y)[0][0]) for y in ys]
        arr = np.stack([ds[v].isel({dim: idx}).transpose(dim, "lat", "lon").values
                        for v in cond_vars], axis=1).astype(np.float32)   # (Ty,C,H,W)
        lat, lon = ds["lat"].values, ds["lon"].values
        cond_arrays[f"cond_{scen}"] = arr
        cond_arrays[f"years_{scen}"] = ys
        print(f"[cond] {scen}: {len(ys)} years {ys[0]}..{ys[-1]}  shape {arr.shape}")
        ds.close()

        # parity: app normalise() vs the training-side functions, last year
        ref = smooth_cond_spatial(arr[-1:].transpose(1, 0, 2, 3), sigma_cfg,
                                  "gaussian", cond_vars)
        ref = normalize_tensor_cond(torch.from_numpy(ref), cond_vars, sigma_cfg)
        ref = ref.numpy()[:, 0]
        app = _app_normalise(arr[-1], sigma_cfg, anchors, cond_vars)
        err = float(np.abs(app - ref).max())
        print(f"[parity] {scen} {ys[-1]}: max |app - training pipeline| = {err:.2e}")
        if err > 1e-4:
            sys.exit("[guard] app normalisation does not match the training pipeline")
    np.savez_compressed(os.path.join(a.out, "cond_base.npz"), lat=lat, lon=lon,
                        **cond_arrays)

    # ── baseline + reference maps from the ens25 eval ───────────────────────
    eval_dir = os.path.expanduser(a.eval_dir)
    base, refs = {}, {}
    for var in target_vars:
        f = os.path.join(eval_dir, f"{var}_hist.nc")
        with xr.open_dataset(f) as ds:
            da = ds[f"{var}_model"]
            base[var] = da.sel(year=slice(*BASELINE)).mean(("member", "year")) \
                          .values.astype(np.float32)
            n_mem = da.sizes["member"]
        print(f"[baseline] {var}: {n_mem} members, global mean "
              f"{float((base[var] * np.cos(np.deg2rad(lat))[:, None]).sum() / (np.cos(np.deg2rad(lat)).sum() * len(lon))):.3f}")
        for scen in SCENARIOS:
            f = os.path.join(eval_dir, f"{var}_{scen}.nc")
            ys = cond_arrays[f"years_{scen}"]
            with xr.open_dataset(f) as ds:
                da = ds[f"{var}_model"]
                have = [y for y in ys if y in set(da["year"].values.tolist())]
                refs[f"{var}_{scen}"] = da.sel(year=have).mean("member").values.astype(np.float32)
                refs[f"years_{scen}"] = np.array(have)
            print(f"[reference] {var} {scen}: {len(have)} years")
    np.savez_compressed(os.path.join(a.out, "baseline.npz"), **base)
    np.savez_compressed(os.path.join(a.out, "reference.npz"), **refs)

    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short",
                                          "HEAD"], text=True).strip()
    except Exception:                                            # noqa: BLE001
        commit = "unknown"
    meta = dict(
        checkpoint=os.path.basename(a.ckpt),
        code_commit=commit,
        n_parameters=n_par,
        model=model_kw,
        scheduler=sched_kw,
        sample_steps=50,
        cond_vars=cond_vars,
        target_vars=target_vars,
        cond_smooth_sigma=sigma_cfg,
        anchors={k: [float(v[0]), float(v[1])] for k, v in anchors.items()},
        target_norm={
            "TREFHT": {"kind": "linear", "scale": 21.0, "offset": 4.5, "units": "degC"},
            "PRECT": {"kind": "log1p", "mean": PRECT_LOG_MEAN, "std": PRECT_LOG_STD,
                      "units": "mm/day"},
        },
        scenarios=list(SCENARIOS),
        baseline=dict(period=list(BASELINE), source=os.path.basename(eval_dir),
                      members=int(n_mem)),
    )
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] bundle in {a.out}")
    for fn in sorted(os.listdir(a.out)):
        print(f"   {fn:16s} {os.path.getsize(os.path.join(a.out, fn)) / 1e6:8.1f} MB")


def _app_normalise(raw, sigma, anchors, cond_vars):
    """Call the app's own Emulator.normalise without building a model."""
    from emulator import Emulator
    e = Emulator.__new__(Emulator)
    e.cond_vars, e.sigma = cond_vars, sigma
    e.anchors = {k: (float(v[0]), float(v[1])) for k, v in anchors.items()}
    return e.normalise(raw)


if __name__ == "__main__":
    main()
