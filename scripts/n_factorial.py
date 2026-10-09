#!/usr/bin/env python3
"""What emissions cause N, the single-forcing non-additivity?

N is defined from the single-forcing runs as

    N = ALL - GHG - AAER [+ PI]
      = f(C,S,B) - f(C,0,0) - f(0,S,B) + f(0,0,0)

In an ANOVA expansion of f over the three conditioning channels this is an
EXACT identity:

    N = CS + CB + CSB

i.e. N is the CO2xSUL term plus the CO2xBC term plus the three-way term, and
contains NO SUL x BC term at all. So "which emissions cause N" is answerable by
running the eight corners of the (CO2, SUL, BC) cube and reading off those
three interactions.

OFF STATES COME FROM THE SINGLE-FORCING COND FILES, not from zeros: GHG holds
SUL and BC at 1850 while CO2 follows the scenario, AAER holds CO2 at 1850. Using
those files as the off-states makes the emulator's N the same object CESM2's
single-forcing runs define. Zeroing a channel instead would be a different
experiment and would not reproduce N.

CLOSURE IS THE TEST THAT MATTERS. N computed from four corners must equal
CS+CB+CSB computed from all eight, to within sampling noise. If it does not,
the decomposition is wrong and nothing downstream is worth running.

Paired seeds throughout: every corner uses the same noise, so the differences
are paired and most sampling noise cancels. The terms are differences of
differences, so without that they are buried.

    python scripts/n_factorial.py --checkpoint ... --members 5
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# DENORM_FN["TREFHT"] = x*21+4.5; every quantity here is a DIFFERENCE, so the
# offset cancels and the slope alone converts to degC.
TREFHT_SCALE = 21.0

FILES = {
    "all":  "emissions_ssp370_only_timefixed_bc_co2fix.nc",   # C on, S on, B on
    "ghg":  "emissions_ghg_only_timefixed_bc_co2fix.nc",      # C on, S/B at 1850
    "aaer": "emissions_aaer_only_timefixed_bc_co2fix.nc",     # C at 1850, S/B on
}


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
    ap.add_argument("--model-config", default="configs/config_aero.yaml")
    ap.add_argument("--data-config", default="configs/config_data_ybias_BCprect_nopca.yaml")
    ap.add_argument("--window", type=int, nargs=2, default=[2041, 2050],
                    help="decade to average; must exist in ghg/aaer (they end 2050)")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--out", default="/scratch/project_462001112/analysis/n_factorial")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import lumi_paths as L
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import eval_aero as EA
    import data.climate_dataset as cd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32                      # model stays fp32; bf16 via autocast
    autocast_dt = torch.bfloat16 if device.type == "cuda" else None
    model, _ = EA.load_model(args.checkpoint, args.model_config, device)
    model = model.to(dtype)
    cfg = L.resolve_cfg(OmegaConf.load(args.model_config))
    scheduler = instantiate(cfg.scheduler)
    dcfg = L.resolve_cfg(OmegaConf.load(args.data_config))
    cond_vars = [str(v) for v in dcfg["cond_vars"]]
    sigmas = [float(s) for s in OmegaConf.to_container(dcfg["cond_smooth_sigma"], resolve=True)]
    out_ch = int(cfg.model.get("out_channels", 1))
    iC, iS, iB = (cond_vars.index(v) for v in ("CO2", "SUL", "BC"))

    src = {}
    for k, f in FILES.items():
        src[k] = read(f"{L.DATA}/{f}", cond_vars)
    w0, w1 = args.window
    yrs = None
    for k, (y, _, _, _) in src.items():
        s = y[(y >= w0) & (y <= w1)]
        yrs = s if yrs is None else np.intersect1d(yrs, s)
    if yrs.size == 0:
        sys.exit(f"[N] no overlap in {w0}-{w1} across {list(FILES)}")
    lat, lon = src["all"][2], src["all"][3]
    idx = {k: np.searchsorted(src[k][0], yrs) for k in src}
    print(f"[N] window {yrs.min()}-{yrs.max()} ({yrs.size} years), "
          f"cond_vars={cond_vars} sigma={sigmas}")

    # Per-channel ON / OFF fields. OFF = the single-forcing file that holds
    # that channel at 1850.
    ON = {iC: src["all"][1][iC][idx["all"]],
          iS: src["all"][1][iS][idx["all"]],
          iB: src["all"][1][iB][idx["all"]]}
    OFF = {iC: src["aaer"][1][iC][idx["aaer"]],    # aaer holds CO2 at 1850
           iS: src["ghg"][1][iS][idx["ghg"]],      # ghg holds SUL at 1850
           iB: src["ghg"][1][iB][idx["ghg"]]}
    for nm, i in (("CO2", iC), ("SUL", iS), ("BC", iB)):
        rel = np.abs(ON[i] - OFF[i]).max() / max(np.abs(ON[i]).max(), 1e-30)
        print(f"[N] {nm}: on-vs-off relative max difference {rel:.3e}")

    def to_cond(arr):
        sm = cd.smooth_cond_spatial(arr.astype(np.float32), sigmas, "gaussian", cond_vars)
        return cd.normalize_tensor_cond(torch.from_numpy(sm), cond_vars, sigmas).contiguous()

    corners = list(itertools.product((0, 1), repeat=3))        # (c, s, b)
    f = {}
    for c, s, b in corners:
        arr = np.empty_like(ON[iC])[None].repeat(len(cond_vars), 0) \
            if False else np.stack([ON[iC] if c else OFF[iC],
                                    ON[iS] if s else OFF[iS],
                                    ON[iB] if b else OFF[iB]])
        ct = to_cond(arr)
        mem = []
        for m in range(args.members):
            y = EA.generate_timeseries(
                model, scheduler, ct, device, dtype,
                sample_steps=args.sample_steps, batch_size=8, seed=1000 + m,
                autocast_dtype=autocast_dt, out_channels=out_ch, target_channel=0)
            mem.append(np.asarray(y, float).mean(0))           # decade mean
        f[(c, s, b)] = np.stack(mem)                            # (M, H, W)
        print(f"  corner C={c} S={s} B={b} done", flush=True)

    M = args.members
    mean = {k: v.mean(0) * TREFHT_SCALE for k, v in f.items()}
    # Per-member spread of the N estimate -> SE of its mean.
    n_mem = np.stack([(f[(1, 1, 1)][m] - f[(1, 0, 0)][m] - f[(0, 1, 1)][m]
                       + f[(0, 0, 0)][m]) * TREFHT_SCALE for m in range(M)])
    se = float(n_mem.std(0).mean()) / np.sqrt(M)

    # ANOVA terms, main effects relative to the all-off corner.
    g = lambda c, s, b: mean[(c, s, b)]
    C   = g(1,0,0) - g(0,0,0)
    S   = g(0,1,0) - g(0,0,0)
    B   = g(0,0,1) - g(0,0,0)
    CS  = g(1,1,0) - g(1,0,0) - g(0,1,0) + g(0,0,0)
    CB  = g(1,0,1) - g(1,0,0) - g(0,0,1) + g(0,0,0)
    SB  = g(0,1,1) - g(0,1,0) - g(0,0,1) + g(0,0,0)
    CSB = (g(1,1,1) - g(1,1,0) - g(1,0,1) - g(0,1,1)
           + g(1,0,0) + g(0,1,0) + g(0,0,1) - g(0,0,0))

    N_direct = g(1,1,1) - g(1,0,0) - g(0,1,1) + g(0,0,0)
    N_anova  = CS + CB + CSB

    wgt = np.cos(np.deg2rad(lat))[:, None]
    gm = lambda a: float(np.average(a, weights=np.broadcast_to(wgt, a.shape)))
    rms = lambda a: float(np.sqrt(np.average(a**2, weights=np.broadcast_to(wgt, a.shape))))

    print(f"\n[N] CLOSURE TEST  (N from 4 corners vs CS+CB+CSB from all 8)")
    print(f"  N direct   global mean {gm(N_direct):+.4f}  rms {rms(N_direct):.4f} degC")
    print(f"  CS+CB+CSB  global mean {gm(N_anova):+.4f}  rms {rms(N_anova):.4f} degC")
    resid = rms(N_direct - N_anova)
    print(f"  closure residual rms   {resid:.5f} degC   "
          f"({100*resid/max(rms(N_direct),1e-30):.2f}% of N)")
    print(f"  SE of the N mean       {se:.5f} degC ({M} paired members)")
    print(f"  -> {'CLOSES (identity holds)' if resid < 3*se else 'DOES NOT CLOSE — decomposition is wrong'}")

    print(f"\n[N] WHAT EMISSIONS CAUSE N")
    print(f"  {'term':>10} {'global mean':>12} {'rms':>9} {'share of N':>12}")
    print("  " + "-"*48)
    for nm, t in (("CO2xSUL", CS), ("CO2xBC", CB), ("3-way", CSB)):
        print(f"  {nm:>10} {gm(t):+12.4f} {rms(t):9.4f} "
              f"{100*rms(t)/max(rms(N_direct),1e-30):11.1f}%")
    print(f"  {'N':>10} {gm(N_direct):+12.4f} {rms(N_direct):9.4f}")
    print(f"\n  for reference (NOT part of N):")
    for nm, t in (("SULxBC", SB), ("CO2 main", C), ("SUL main", S), ("BC main", B)):
        print(f"  {nm:>10} {gm(t):+12.4f} {rms(t):9.4f}")

    np.savez_compressed(
        os.path.join(args.out, f"n_factorial_{yrs.min()}_{yrs.max()}.npz"),
        lat=lat, lon=lon, years=yrs, members=M, se=se,
        N_direct=N_direct, N_anova=N_anova,
        CS=CS, CB=CB, CSB=CSB, SB=SB, C=C, S=S, B=B,
        **{f"corner_{c}{s}{b}": mean[(c, s, b)] for c, s, b in corners})
    print(f"\n[N] wrote {args.out}/n_factorial_{yrs.min()}_{yrs.max()}.npz")


if __name__ == "__main__":
    main()
