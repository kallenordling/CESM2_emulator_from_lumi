#!/usr/bin/env python3
"""Does perturbing ONE region's aerosol emissions move climate THERE?

The premise behind regional conditioning is that local emissions have a local
effect. This measures it directly instead of inferring it from global-mean
skill: scale SUL and BC inside one region box, leave the rest of the world
alone, and look at where the emulator's response lands.

WHY THIS IS POSSIBLE NOW. It was not, before 2026-10-01. The PCA bottleneck
destroyed 91-95% of a regional perturbation and smeared the remainder across
other regions, which is why attribution had to be made to EOF modes rather than
to places. The cleanup removed PCA entirely -- the pipeline is now
smooth -> min/max normalise -- so a regional perturbation reaches the model
intact apart from the smoothing kernel.

THE PERTURBATION IS APPLIED TO THE RAW INVENTORY, before smoothing and
normalisation, so it travels the identical path the model trained on. The
anchors come from the checkpoint (COND_PROCESSED_NORM), so baseline and
perturbed are normalised with the SAME lo/hi -- otherwise the perturbation
would move the anchors and the difference would conflate a regional change with
a global rescaling.

PAIRED SEEDS. Diffusion sampling noise is large next to a regional aerosol
signal, so each member is sampled with the SAME seed for baseline and
perturbed. The difference is then a paired comparison and most of the noise
cancels. Without this the signal is buried; an unpaired version of this test
would mostly measure the sampler.

WHAT IT CAN AND CANNOT SAY. It measures the MODEL's sensitivity, not whether
that sensitivity is right -- there is no CESM2 run perturbing these same boxes
(RAMIP's ssp370-126aer is a global aerosol change, not a regional one). Read it
as "the emulator responds locally", never as "the emulator responds correctly".
A large perturbation is also out of hull: zeroing East China while the rest of
the world follows ssp370 is a forcing combination never trained on, which is
where this model is already known to fail. --scale 0.5 stays nearer the hull;
--scale 0.0 is the removal case and should be read separately.

    python scripts/regional_perturbation.py \\
        --checkpoint /scratch/.../run_mmlin_co2fix_630.pt \\
        --region east_china --scale 0.5 --members 5
"""
import argparse
import os
import sys

import numpy as np
import torch
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# lat0, lat1, lon0, lon1  (lon in -180..180)
REGIONS = {
    "east_china": (22.0, 42.0, 105.0, 123.0),
    "india":      (8.0, 30.0, 70.0, 88.0),
    "europe":     (40.0, 60.0, -10.0, 30.0),
    "eastern_us": (30.0, 45.0, -95.0, -70.0),
}
AEROSOL = ("SUL", "BC")


def region_mask(lat, lon180, box):
    la0, la1, lo0, lo1 = box
    m = ((lat[:, None] >= la0) & (lat[:, None] <= la1) &
         (lon180[None, :] >= lo0) & (lon180[None, :] <= lo1))
    return m


def great_circle_km(lat1, lon1, lat2, lon2):
    """Distance from one point to a lat/lon grid, in km."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="configs/config_aero.yaml")
    ap.add_argument("--data-config", default="configs/config_data_ybias_BCprect_nopca.yaml")
    ap.add_argument("--cond-file", default=None,
                    help="default: the ssp370 _co2fix cond file under L.DATA")
    ap.add_argument("--region", default="east_china", choices=sorted(REGIONS))
    ap.add_argument("--scale", type=float, default=0.5,
                    help="multiply SUL and BC inside the box by this (0 = removal)")
    ap.add_argument("--year", type=int, default=2050)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--out", default="plots/regional_perturbation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import lumi_paths as L
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import eval_aero as EA
    import data.climate_dataset as cd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model stays float32 and bf16 is applied through AUTOCAST only. Casting
    # the whole model to bf16 crashes with "Input type (float) and bias type
    # (c10::BFloat16) should be the same" when MIOpen falls back silently --
    # the failure eval_aero.py:2161 already documents, and which this script
    # hit on its first run.
    dtype = torch.float32
    autocast_dt = torch.bfloat16 if device.type == "cuda" else None
    model, _ = EA.load_model(args.checkpoint, args.model_config, device)
    model = model.to(dtype)
    cfg = L.resolve_cfg(OmegaConf.load(args.model_config))
    scheduler = instantiate(cfg.scheduler)
    data_cfg = L.resolve_cfg(OmegaConf.load(args.data_config))
    cond_vars = [str(v) for v in data_cfg["cond_vars"]]
    sig = OmegaConf.to_container(data_cfg["cond_smooth_sigma"], resolve=True)
    sigmas = [float(s) for s in sig]
    print(f"[perturb] cond_vars={cond_vars} sigma={sigmas}")

    cond_file = args.cond_file or f"{L.DATA}/emissions_ssp370_only_timefixed_bc_co2fix.nc"
    raw = xr.open_dataset(cond_file)
    tdim = "time" if "time" in raw.dims else "year"
    years = np.asarray(raw[tdim].values).astype(int)
    yi = int(np.argmin(np.abs(years - args.year)))
    lat = raw["lat"].values
    lon = raw["lon"].values
    lon180 = ((lon + 180) % 360) - 180
    base_np = np.stack([np.asarray(raw[v].values, np.float64) for v in cond_vars])
    raw.close()

    box = REGIONS[args.region]
    mask = region_mask(lat, lon180, box)
    print(f"[perturb] region={args.region} box={box} cells={int(mask.sum())}")

    pert_np = base_np.copy()
    for vi, v in enumerate(cond_vars):
        if v in AEROSOL:
            pert_np[vi][:, mask] *= args.scale
    moved = float(np.abs(pert_np - base_np).sum())
    print(f"[perturb] scale={args.scale}: total |delta emissions| = {moved:.4g}")
    if moved == 0:
        sys.exit("[perturb] perturbation is a no-op — wrong region or scale=1")

    def to_cond(arr):
        """Raw (n_var, T, H, W) -> normalised tensor, via the training path.

        The checkpoint's anchors are already injected by load_model, so both
        calls use the SAME lo/hi and the difference isolates the region.
        """
        sm = cd.smooth_cond_spatial(arr.astype(np.float32), sigmas, "gaussian", cond_vars)
        return cd.normalize_tensor_cond(torch.from_numpy(sm), cond_vars, sigmas).contiguous()

    cond_b = to_cond(base_np)
    cond_p = to_cond(pert_np)
    d_cond = (cond_p - cond_b)[:, yi].numpy()
    print(f"[perturb] cond delta: max|d| per channel = "
          + ", ".join(f"{v}={np.abs(d_cond[i]).max():.4f}"
                      for i, v in enumerate(cond_vars)))

    # One year is enough: the response to a single-year forcing change is what
    # locality means here. Slice both to the same year window.
    sl = slice(yi, yi + 1)
    out = {}
    for tag, ct in (("base", cond_b), ("pert", cond_p)):
        mem = []
        for m in range(args.members):
            # SAME seed per member index for base and pert -> paired noise.
            y = EA.generate_timeseries(
                model, scheduler, ct[:, sl], device, dtype,
                sample_steps=args.sample_steps, batch_size=1, seed=1000 + m,
                autocast_dtype=autocast_dt,
                out_channels=int(cfg.model.get("out_channels", 1)),
                target_channel=0)
            mem.append(np.asarray(y, float))
            print(f"  [{tag}] member {m+1}/{args.members}", flush=True)
        out[tag] = np.stack(mem)                       # (M, T, H, W)

    resp = out["pert"].mean(0)[0] - out["base"].mean(0)[0]        # (H, W)
    # Per-member spread of the paired difference: the noise floor this test
    # actually has, which is what decides whether the signal is real.
    per_member = out["pert"][:, 0] - out["base"][:, 0]
    # std ACROSS members is a property of the sampler and does NOT shrink as
    # members are added -- it barely moved from 5 members (0.0028) to 25
    # (0.0030). The yardstick for the MEAN response is the standard error,
    # std/sqrt(M). Reporting the per-member std as "noise" understated the
    # significance of every region in the first 25-member run.
    spread = float(per_member.std(0).mean())
    noise = spread / max(len(per_member), 1) ** 0.5

    w = np.cos(np.deg2rad(lat))[:, None]
    # Halving a region's aerosols causes REAL global-mean warming, and that
    # component is spatially uniform. Left in, it inflates |response|
    # everywhere and buries any regional pattern -- the locality metric would
    # then mostly measure the global-mean shift. So locality is measured on the
    # PATTERN (response minus its area-weighted global mean) and the global
    # mean is reported separately, since it is a real part of the answer.
    gmean = float(np.average(resp, weights=np.broadcast_to(w, resp.shape)))
    pattern = resp - gmean
    print(f"  global-mean shift        {gmean:+.4f} degC  (uniform component)")
    print(f"  pattern amplitude        {np.abs(pattern).mean():.4f} degC")
    amag = np.abs(pattern) * w
    tot = amag.sum()
    clat = 0.5 * (box[0] + box[1]); clon = 0.5 * (box[2] + box[3])
    dist = great_circle_km(clat, clon, lat[:, None], lon180[None, :])

    inside = amag[mask].sum() / tot * 100
    rings = {f"<{r}km": amag[dist <= r].sum() / tot * 100 for r in (500, 1000, 2000, 4000)}
    order = np.argsort(dist.ravel())
    cum = np.cumsum(amag.ravel()[order]) / tot
    r50 = float(dist.ravel()[order][int(np.searchsorted(cum, 0.5))])

    print(f"\n[perturb] RESPONSE to {args.region} aerosols x{args.scale} "
          f"({args.year}, {args.members} paired members)")
    print(f"  mean |response|          {np.abs(resp).mean():.4f} degC")
    # (locality numbers below are on the PATTERN, global mean removed)
    print(f"  per-member spread        {spread:.4f} degC  (sampler, does not "
          f"shrink with M)")
    print(f"  standard error of mean   {noise:.5f} degC  (spread/sqrt({len(per_member)}))")
    print(f"  inside the box           {inside:.1f}% of |response|")
    for k, v in rings.items():
        print(f"  within {k:<8s}          {v:.1f}%")
    print(f"  radius holding 50%       {r50:.0f} km")
    print(f"  pattern / SE             {np.abs(pattern).mean()/max(noise,1e-9):.1f}"
          f"   (is the PATTERN resolved?)")
    # A uniform response would put this fraction inside the box; anything near
    # it means the model is NOT responding locally.
    unif = (w * np.ones_like(resp))[mask].sum() / (w * np.ones_like(resp)).sum() * 100
    print(f"  (box is {unif:.2f}% of global area — a uniform response would "
          f"score that, so {inside/max(unif,1e-9):.1f}x = locality factor)")

    np.savez_compressed(
        os.path.join(args.out, f"resp_{args.region}_x{args.scale}_{args.year}.npz"),
        response=resp, pattern=pattern, gmean=gmean,
        lat=lat, lon=lon, box=np.array(box), inside=inside,
        r50=r50, noise=noise, rings=np.array(list(rings.values())),
        ring_names=np.array(list(rings), dtype=object))
    print(f"[perturb] wrote {args.out}/resp_{args.region}_x{args.scale}_{args.year}.npz")


if __name__ == "__main__":
    main()
