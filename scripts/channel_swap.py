#!/usr/bin/env python3
"""Which conditioning CHANNEL carries the model's response to a scenario?

Analysis B. A channel swap changes the physical scenario, so a hybrid run
cannot be scored against the test scenario's truth. What it CAN do is
decompose the model's own response relative to a reference scenario:

    R(X)  =  model(X) - model(ssp370)                     the model's response
    R(X) ~=  sum_c [ model(ssp370 with channel c from X) - model(ssp370) ]

each term being one channel's contribution, with the residual measuring
non-additivity. Since the model's TOTAL response is known to be too weak --
the aerosol-removal signal falls from 0.98 K (CESM2) to 0.36 K by 2070 -- this
says WHICH channel the missing response belongs to.

ADDITIVITY IS MEASURED, NOT ASSUMED. The emulator's response is known not to
be GHG+AAER, so a residual is expected; it is reported as its own term rather
than folded into the channels.

Every run shares the checkpoint's anchors and uses PAIRED SEEDS, so the
differences are paired and most sampling noise cancels. Without that the
channel contributions are buried: they are differences of large fields.

    python scripts/channel_swap.py --checkpoint ... --scenario ssp370-126aer
"""
import argparse
import os
import sys

import numpy as np
import torch
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

SCEN_FILE = {
    "ssp370":        "emissions_ssp370_only_timefixed_bc_co2fix.nc",
    "ssp370-126aer": "emissions_ssp370co2_ssp126aer_bc_2015-2079_co2fix.nc",
    "ssp126":        "emissions_ssp126_only_timefixed_bc_co2fix.nc",
    "ssp245":        "emissions_ssp245_only_timefixed_bc_co2fix.nc",
}
WINDOW = {"ssp370-126aer": (2070, 2079), "ssp126": (2091, 2100),
          "ssp245": (2091, 2100)}


def read(path, cond_vars):
    ds = xr.open_dataset(path)
    td = "time" if "time" in ds.dims else "year"
    y = np.asarray(ds[td].values).astype(int)
    a = np.stack([np.asarray(ds[v].values, np.float64) for v in cond_vars])
    lat, lon = ds["lat"].values, ds["lon"].values
    ds.close()
    return y, a, lat, lon


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scenario", default="ssp370-126aer", choices=sorted(WINDOW))
    ap.add_argument("--reference", default="ssp370")
    ap.add_argument("--model-config", default="configs/config_aero.yaml")
    ap.add_argument("--data-config", default="configs/config_data_ybias_BCprect_nopca.yaml")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--out", default="/scratch/project_462001112/analysis/channel_swap")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import lumi_paths as L
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import eval_aero as EA
    import data.climate_dataset as cd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp32 model + bf16 autocast; casting the model to bf16 crashes MIOpen
    # (eval_aero.py:2161).
    dtype, autocast_dt = torch.float32, (torch.bfloat16 if device.type == "cuda" else None)
    model, _ = EA.load_model(args.checkpoint, args.model_config, device)
    model = model.to(dtype)
    cfg = L.resolve_cfg(OmegaConf.load(args.model_config))
    scheduler = instantiate(cfg.scheduler)
    dcfg = L.resolve_cfg(OmegaConf.load(args.data_config))
    cond_vars = [str(v) for v in dcfg["cond_vars"]]
    sigmas = [float(s) for s in OmegaConf.to_container(dcfg["cond_smooth_sigma"], resolve=True)]
    out_ch = int(cfg.model.get("out_channels", 1))
    print(f"[swap] cond_vars={cond_vars} sigma={sigmas}")

    yr, ar, lat, lon = read(f"{L.DATA}/{SCEN_FILE[args.reference]}", cond_vars)
    yx, ax, _, _ = read(f"{L.DATA}/{SCEN_FILE[args.scenario]}", cond_vars)
    w0, w1 = WINDOW[args.scenario]
    # Common years in the comparison window: the hybrid must be evaluated on
    # the SAME years as the reference or the difference includes a time shift.
    yrs = np.intersect1d(yr[(yr >= w0) & (yr <= w1)], yx[(yx >= w0) & (yx <= w1)])
    if yrs.size == 0:
        sys.exit(f"[swap] no overlap in {w0}-{w1} between {args.reference} and {args.scenario}")
    ir = np.searchsorted(yr, yrs); ix = np.searchsorted(yx, yrs)
    print(f"[swap] window {yrs.min()}-{yrs.max()} ({yrs.size} years)")

    # Which channels actually differ? A channel that is identical needs no run.
    changed = []
    for i, v in enumerate(cond_vars):
        rel = np.abs(ax[i][ix] - ar[i][ir]).max() / max(np.abs(ar[i][ir]).max(), 1e-30)
        print(f"[swap] {v}: relative max difference {rel:.3e}"
              f"{'  (identical, skipped)' if rel < 1e-10 else ''}")
        if rel >= 1e-10:
            changed.append(v)
    if not changed:
        sys.exit("[swap] scenarios are identical in every channel")

    def to_cond(arr):
        sm = cd.smooth_cond_spatial(arr.astype(np.float32), sigmas, "gaussian", cond_vars)
        return cd.normalize_tensor_cond(torch.from_numpy(sm), cond_vars, sigmas).contiguous()

    runs = {"reference": ar[:, ir]}
    for v in changed:                       # one hybrid per changed channel
        h = ar[:, ir].copy()
        h[cond_vars.index(v)] = ax[cond_vars.index(v)][ix]
        runs[f"swap_{v}"] = h
    runs["full"] = ax[:, ix]

    sampled = {}
    for tag, arr in runs.items():
        ct = to_cond(arr)
        mem = []
        for m in range(args.members):
            y = EA.generate_timeseries(
                model, scheduler, ct, device, dtype,
                sample_steps=args.sample_steps, batch_size=8, seed=1000 + m,
                autocast_dtype=autocast_dt, out_channels=out_ch, target_channel=0)
            mem.append(np.asarray(y, float).mean(0))     # decade mean
            print(f"  [{tag}] member {m+1}/{args.members}", flush=True)
        sampled[tag] = np.stack(mem)
        print(f"[swap] {tag} done", flush=True)

    w = np.cos(np.deg2rad(lat))[:, None]
    gm = lambda a: float(np.average(a, weights=np.broadcast_to(w, a.shape)))
    ref = sampled["reference"].mean(0)
    full = sampled["full"].mean(0) - ref
    parts = {v: sampled[f"swap_{v}"].mean(0) - ref for v in changed}
    resid = full - sum(parts.values())
    spread = float((sampled["full"][:, ] - sampled["reference"]).std(0).mean())
    se = spread / np.sqrt(args.members)

    print(f"\n[swap] RESPONSE of {args.scenario} relative to {args.reference}")
    print(f"       ({args.members} paired members, SE of the mean {se:.4f} degC)")
    print(f"  {'term':>14} {'global mean':>12} {'rms':>9} {'share of |full|':>16}")
    print("  " + "-"*56)
    fr = float(np.sqrt(np.average(full**2, weights=np.broadcast_to(w, full.shape))))
    for v, p in parts.items():
        pr = float(np.sqrt(np.average(p**2, weights=np.broadcast_to(w, p.shape))))
        print(f"  {v:>14} {gm(p):+12.4f} {pr:9.4f} {100*pr/max(fr,1e-30):15.1f}%")
    rr = float(np.sqrt(np.average(resid**2, weights=np.broadcast_to(w, resid.shape))))
    print(f"  {'interaction':>14} {gm(resid):+12.4f} {rr:9.4f} {100*rr/max(fr,1e-30):15.1f}%")
    print(f"  {'FULL':>14} {gm(full):+12.4f} {fr:9.4f}")
    add = 100 * rr / max(fr, 1e-30)
    print(f"\n  additivity: residual is {add:.1f}% of the full response "
          f"({'additive' if add < 15 else 'NOT additive — channels interact'})")

    np.savez_compressed(
        os.path.join(args.out, f"swap_{args.scenario}_vs_{args.reference}.npz"),
        lat=lat, lon=lon, years=yrs, full=full, residual=resid,
        se=se, members=args.members,
        **{f"part_{v}": p for v, p in parts.items()})
    print(f"[swap] wrote {args.out}/swap_{args.scenario}_vs_{args.reference}.npz")


if __name__ == "__main__":
    main()
