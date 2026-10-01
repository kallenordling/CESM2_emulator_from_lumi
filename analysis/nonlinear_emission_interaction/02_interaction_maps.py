#!/usr/bin/env python3
"""
================================================================================
 SPEC FIGURES 2-4 — THE NONLINEAR RESPONSE MAP, AND THE EMISSION-SENSITIVITY MAPS
================================================================================

    python analysis/nonlinear_emission_interaction/02_interaction_maps.py \
        <checkpoint.pt> --year 2040

Milestone 01 reduced everything to a global-mean scalar, so it produced no maps.
This adds the two spatial objects the spec asks for, which are different things
and must not be conflated:

  RESPONSE MAP (spec Fig 2)      N_T(r) over the output grid
      = f(C,A) - f(C,0) - f(0,A) + f(0,0), evaluated per output cell.
      Four forward passes, no gradients. WHERE the nonlinear response appears.

  SOURCE MAPS (spec Figs 3, 4)   d2 T(r) / dC dA(x) over the INPUT grid
      C stays a scalar — the fraction of the CO2 forcing — while the aerosol
      channels stay a full field, so one double-backward gives the sensitivity
      to every aerosol emission cell at once. WHERE the emissions that drive the
      interaction are. Computed separately for SO4 and BC, as section 3 requires,
      and never summed.

The two spatial axes are deliberately kept apart (spec section 13): the response
map is indexed by r, the source maps by x.

WHAT THE SOURCE MAP IS NOT
--------------------------
It is a derivative with respect to the CONDITIONING FIELD, which is what the
network consumes. It is not a licence to say "cutting emissions in this cell
changes N_T by that much": conditioning is PCA-denoised to five modes per
aerosol species, and a single-cell emission change survives that projection at
about 9% of its variance, with over 90% of the remainder landing elsewhere.
Read the large-scale organisation, and make quantitative claims in mode space.

UNITS
-----
Normalised model units. The cond regrid deflates the species by roughly 4.7x
without conserving sums, so K/(GtCO2 x Mt) would carry a factor the pipeline
cannot support. Rankings and ratios are meaningful; absolute sensitivities are
not.
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from omegaconf import OmegaConf
from hydra.utils import instantiate
import lumi_paths as L  # noqa: F401
from eval_aero import load_model, build_cond_tensor

COND_DIR = "/scratch/project_462001328/emulator_data"

# Response regions the source maps are computed FOR (spec section 13's r).
RESPONSE_REGIONS = [
    ("Global",      -90.0,  90.0,   0.0, 360.0),
    ("Arctic",       66.5,  90.0,   0.0, 360.0),
    ("N. Atlantic",  45.0,  65.0, 300.0, 350.0),
    ("Tropics",     -20.0,  20.0,   0.0, 360.0),
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
parser.add_argument("--model-config", default="configs/config_aero.yaml")
parser.add_argument("--data-config", default="configs/config_data_ybias_BCprect.yaml")
parser.add_argument("--cond-file",
                    default=f"{COND_DIR}/emissions_ssp370_only_timefixed_bc_co2fix.nc")
parser.add_argument("--baseline-file",
                    default=f"{COND_DIR}/emissions_hist_only_timefixed_bc_co2fix.nc")
parser.add_argument("--integrate-grid", type=int, default=0, metavar="N",
                    help="INTEGRATE the source maps over the unit square with "
                         "an NxN Gauss-Legendre rule instead of evaluating "
                         "them at alpha=beta=1. Since A(beta) = A0 + beta*delta, "
                         "the chain rule gives "
                         "N = sum_x delta(x) * INTEGRAL d2T/dalpha dA(x), so "
                         "only the INTEGRATED map decomposes N; the map at "
                         "(1,1) is a local sensitivity and does not sum to it "
                         "(measured: -0.06 K against N = +0.60 K). Costs N*N "
                         "double-backwards per region per seed.")
parser.add_argument("--co2-source", action="store_true",
                    help="also compute the symmetric map "
                         "d2T/dC(x) dbeta -- CO2 as the field, the "
                         "aerosol as the scalar. The default maps "
                         "spatialise the AEROSOL side only, which is "
                         "a choice from the spec, not a property of "
                         "the interaction.")
parser.add_argument("--year", type=int, default=2040)
parser.add_argument("--baseline-year", type=int, default=1850)
parser.add_argument("--cond-vars", default="CO2,SUL,BC")
parser.add_argument("--smooth-sigma", default="0,2,2")
parser.add_argument("--channel", type=int, default=0)
parser.add_argument("--t", type=float, default=1.0,
                    help="diffusion TIME in [0,1]; log_snr is derived "
                         "from it. t=1 is the noisiest end "
                         "(log_snr=-10.0001), t=0.949 is log_snr=-9.")
parser.add_argument("--seeds", default="0,1,2")
parser.add_argument("--out", default=None)
args = parser.parse_args()

cond_vars = args.cond_vars.split(",")
sigmas = [float(s) for s in args.smooth_sigma.split(",")]
CO2_CH = [i for i, v in enumerate(cond_vars) if v.upper() == "CO2"]
AER_CH = [i for i, v in enumerate(cond_vars) if v.upper() in ("SUL", "BC")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Restore the checkpoint's CONDITIONING SETTINGS before anything builds a
# cond tensor. This script calls build_cond_tensor directly, bypassing
# eval_aero.main() where that restoration lives, so without this it silently
# normalises with the DEFAULTS (v1, normalize_first, all4 anchors). On a
# minmax/normalize_last checkpoint that is a different input from the one the
# model trained on -- the same class of silent mismatch that invalidated the
# asinh99 evals and the first normlast evals.
_ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
from data.climate_dataset import set_anchor_scenarios, set_processed_minmax_override
_anc = _ck.get("COND_ANCHORS")
if _anc and _anc != "all4":
    set_anchor_scenarios(_anc)
if _ck.get("COND_PROCESSED_NORM"):
    set_processed_minmax_override(_ck["COND_PROCESSED_NORM"])
print(f"[INTMAPS-COND] anchors={_anc or 'all4'} "
      f"processed={'ckpt' if _ck.get('COND_PROCESSED_NORM') else 'REFIT'}", flush=True)
del _ck

model, _ = load_model(args.checkpoint, args.model_config, device)
for p in model.parameters():
    p.requires_grad_(False)
cfg = L.resolve_cfg(OmegaConf.load(args.model_config))
scheduler = instantiate(cfg.scheduler)
out_channels = int(cfg.model.get("out_channels", 1))
tensor, years, lat, lon = build_cond_tensor(
    args.cond_file, cond_vars, "time",
    cond_smooth_sigma=sigmas, cond_smooth_method="gaussian")
base_tensor, base_years, _, _ = build_cond_tensor(
    args.baseline_file, cond_vars, "time",
    cond_smooth_sigma=sigmas, cond_smooth_method="gaussian")
state = tensor[:, int(np.where(years == args.year)[0][0])]
base = base_tensor[:, int(np.where(base_years == args.baseline_year)[0][0])]
state = state.unsqueeze(0).unsqueeze(2).to(device).float()
base = base.unsqueeze(0).unsqueeze(2).to(device).float()
delta = state - base
_, _, _, H, W = state.shape

lat_t = torch.as_tensor(lat, dtype=torch.float32, device=device)
lon_t = torch.as_tensor(lon, dtype=torch.float32, device=device)
lon_grid, lat_grid = torch.meshgrid(lon_t, lat_t, indexing="xy")
cos_lat = torch.cos(torch.deg2rad(lat_t))[:, None].expand(H, W)
# NOTE: predict_start_from_v takes a TIME t and converts it to log_snr itself
# (continuous_ddpm.py:116). Passing log_snr there double-converts. The model
# wants log_snr, the scheduler wants t -- exactly as unetTrainer.py:1160-1162.
# Parameterise by t: the schedule spans log_snr(0)=+9.21 to log_snr(1)=-10.0001,
# so log_snr=-12 is not reachable at all (t=1.095).
t_diff  = torch.full((1,), args.t, device=device)
log_snr = scheduler.log_snr(t_diff)
print(f"[noise] t = {args.t:.4f}  ->  log_snr = {float(log_snr[0]):+.4f}")


def field(cond_input, noise):
    """The predicted clean field — (H, W), NOT reduced to a scalar."""
    v = model(noise, log_snr, cond_map=cond_input)
    x0 = scheduler.predict_start_from_v(noise, t_diff, v)
    return x0[0, args.channel, 0]


def corner(a, b, noise):
    parts = [base[:, c:c + 1] + (a if c in CO2_CH else b) * delta[:, c:c + 1]
             for c in range(len(cond_vars))]
    return field(torch.cat(parts, dim=1), noise)


def region_weights(lat_min, lat_max, lon_min, lon_max):
    inside = ((lat_grid >= lat_min) & (lat_grid <= lat_max)
              & (lon_grid >= lon_min) & (lon_grid <= lon_max))
    w = torch.where(inside, cos_lat, torch.zeros_like(cos_lat))
    return w / w.sum()


seeds = [int(s) for s in args.seeds.split(",")]
response_maps, source_maps = [], {r[0]: [] for r in RESPONSE_REGIONS}
co2_maps = {r[0]: [] for r in RESPONSE_REGIONS}

for seed in seeds:
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(1, out_channels, 1, H, W, device=device, generator=gen)

    # ── spec Figure 2 — the response map ────────────────────────────────────
    with torch.no_grad():
        n_map = (corner(1, 1, noise) - corner(1, 0, noise)
                 - corner(0, 1, noise) + corner(0, 0, noise))
    response_maps.append(n_map.cpu().numpy())

    # ── spec Figures 3 and 4 — the source maps ──────────────────────────────
    # alpha stays a SCALAR (the CO2 fraction); the aerosol channels stay a FULL
    # FIELD with requires_grad. One double-backward then yields
    # d2 T(region) / d alpha d A(x) for every aerosol cell at once — the whole
    # point of not making beta a scalar here.
    def source_map_at(a, b, weights):
        """d2 T(region) / d alpha d A(x) at the point (alpha, beta) = (a, b)."""
        ta = torch.tensor(float(a), device=device, requires_grad=True)
        aerosol = (base[:, AER_CH] + b * delta[:, AER_CH]).detach().requires_grad_(True)
        parts, ai = [], 0
        for c in range(len(cond_vars)):
            if c in CO2_CH:
                parts.append(base[:, c:c + 1] + ta * delta[:, c:c + 1])
            else:
                parts.append(aerosol[:, ai:ai + 1]); ai += 1
        value = (field(torch.cat(parts, dim=1), noise) * weights).sum()
        g_alpha, = torch.autograd.grad(value, ta, create_graph=True)
        d2, = torch.autograd.grad(g_alpha, aerosol)
        return d2[0, :, 0]

    for name, *box in RESPONSE_REGIONS:
        weights = region_weights(*box)
        if args.integrate_grid:
            # G(x) = INTEGRAL_0^1 INTEGRAL_0^1 d2T/dalpha dA(x). This is the
            # map that decomposes N: sum_x G(x) delta(x) = N exactly, because
            # d2T/dalpha dbeta = sum_x d2T/dalpha dA(x) * delta(x).
            nod, gw = np.polynomial.legendre.leggauss(args.integrate_grid)
            nod = 0.5 * (nod + 1.0)
            gw = 0.5 * gw
            acc = None
            for i, a in enumerate(nod):
                for j, b in enumerate(nod):
                    d2 = source_map_at(a, b, weights) * float(gw[i] * gw[j])
                    acc = d2 if acc is None else acc + d2
                    del d2
            source_maps[name].append(acc.cpu().numpy())
            del acc
        else:
            source_maps[name].append(source_map_at(1.0, 1.0, weights).cpu().numpy())

        # ── the SYMMETRIC object: CO2 as the field, aerosol as the scalar ────
        # The map above spatialises the aerosol side only, because the spec
        # asked which AEROSOL sources drive the interaction. That is a choice,
        # not a property of the interaction: d2T/dC(x) dbeta is equally well
        # defined and answers "where is the CO2 that does it".
        #
        # Two things make it weaker, and both should be read on the figure:
        # CO2 is well mixed, so "where it was emitted" carries far less
        # physical meaning than for aerosol; and CO2 is smoothed with sigma=0
        # (SUL/BC use 2), so this map inherits the single-cell speckle visible
        # in the CO2 row of the EOF figure.
        if args.co2_source:
            tb = torch.tensor(1.0, device=device, requires_grad=True)
            co2 = (base[:, CO2_CH] + delta[:, CO2_CH]).detach().requires_grad_(True)
            parts2, ci = [], 0
            for c in range(len(cond_vars)):
                if c in CO2_CH:
                    parts2.append(co2[:, ci:ci + 1]); ci += 1
                else:
                    parts2.append(base[:, c:c + 1] + tb * delta[:, c:c + 1])
            value2 = (field(torch.cat(parts2, dim=1), noise) * weights).sum()
            g_beta, = torch.autograd.grad(value2, tb, create_graph=True)
            d2c, = torch.autograd.grad(g_beta, co2)
            co2_maps[name].append(d2c[0, :, 0].cpu().numpy())  # (n_co2, H, W)
    print(f"[seed {seed}] response map and {len(RESPONSE_REGIONS)} source maps done",
          flush=True)

out = args.out or os.path.join(_HERE, "results", f"interaction_maps_{args.year}.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
payload = {"lat": lat, "lon": lon, "year": args.year, "seeds": np.array(seeds),
           "integrate_grid": args.integrate_grid,
           "source_map_kind": ("integrated over [0,1]^2 — decomposes N"
                               if args.integrate_grid
                               else "local at alpha=beta=1 — does NOT sum to N"),
           "aerosol_names": np.array([cond_vars[c] for c in AER_CH], dtype=object),
           "response_mean": np.mean(response_maps, axis=0),
           "response_sd": np.std(response_maps, axis=0, ddof=1) if len(seeds) > 1
                          else np.zeros_like(response_maps[0]),
           "regions": np.array([r[0] for r in RESPONSE_REGIONS], dtype=object)}
for name in co2_maps:
    if not co2_maps[name]:
        continue
    st = np.stack(co2_maps[name])                 # (seed, n_co2, H, W)
    payload[f"co2_source_mean_{name}"] = st.mean(axis=0)
    payload[f"co2_source_sd_{name}"] = (st.std(axis=0, ddof=1) if len(seeds) > 1
                                        else np.zeros_like(st[0]))

for name in source_maps:
    stack = np.stack(source_maps[name])           # (seed, n_aer, H, W)
    payload[f"source_mean_{name}"] = stack.mean(axis=0)
    payload[f"source_sd_{name}"] = (stack.std(axis=0, ddof=1) if len(seeds) > 1
                                    else np.zeros_like(stack[0]))

# ── REDUCE ONTO EOF MODES, not source regions ───────────────────────────────
# The full-field source map d2T(region)/dC dA(x) is defined at every cell, but
# the model CANNOT BE DRIVEN at every cell: the conditioning passes through a
# rank-k PCA, so only the span of the retained EOFs is reachable. Measured
# fraction of ||g||^2 lying inside that span, on the existing ep0863 maps:
#
#     region        SUL     BC          (chance for a random vector in
#     Global       1.11%   0.59%         55296 dims is 5/55296 = 0.009%)
#     Arctic       0.88%   0.41%
#     N.Atlantic   0.41%   0.01%   <-- AT CHANCE: that panel is noise
#     Tropics      1.12%   0.49%
#
# So global SUL/BC sit 45-120x above chance and are real, but ~99% of every
# source map is gradient along directions no conditioning file can produce.
# Plotting them as geographic "source regions" attributes signal to places the
# model cannot be perturbed. The directional derivative along EOF_k IS the
# projection below, so this needs no extra model evaluation -- the old
# reduction was simply the wrong one.
#
# The conditioning DISPLACEMENT the maps are integrated along, per aerosol
# channel, exactly as the model received it (normalised, smoothed, projected).
# Saved so the attribution density G(x) * delta(x) can be formed from this file
# alone. It is also the quantity a clipping transform distorts: under v1 a cell
# pinned at +1 in 2040 and at -1 in 1850 swings the full range whatever its
# emissions, so a minor emitter such as the Arabian Peninsula gets the same
# delta as East China. Comparing delta across transforms is the saturation test.
# state/base are (batch=1, vars, time=1, H, W) by this point, hence the indexing.
payload["delta"] = (state - base)[0, AER_CH, 0].detach().cpu().numpy().astype(np.float32)
payload["state_aer"] = state[0, AER_CH, 0].detach().cpu().numpy().astype(np.float32)
payload["base_aer"] = base[0, AER_CH, 0].detach().cpu().numpy().astype(np.float32)
payload["checkpoint"] = np.array(os.path.abspath(args.checkpoint))
try:
    from data.climate_dataset import get_active_cond_transform
    payload["cond_transform"] = np.array(get_active_cond_transform())
    print(f"[maps] conditioning transform in force: {get_active_cond_transform()}")
except ImportError:
    payload["cond_transform"] = np.array("unknown")

np.savez_compressed(out, **payload, allow_pickle=True)

area = np.cos(np.deg2rad(lat))[:, None]
print(f"\n[maps] N_T response map: global mean "
      f"{np.average(payload['response_mean'], weights=np.broadcast_to(area, (H, W))):+.5f}, "
      f"|N_T| {np.average(np.abs(payload['response_mean']), weights=np.broadcast_to(area, (H, W))):.5f}")
for name in source_maps:
    for i, sp in enumerate(payload["aerosol_names"]):
        m = payload[f"source_mean_{name}"][i]
        print(f"[maps] d2T({name})/dC dA[{sp}]: sum {m.sum():+.5f}, "
              f"|.| {np.abs(m).sum():.5f}, "
              f"seed sd/|.| {payload[f'source_sd_{name}'][i].mean() / (np.abs(m).mean() + 1e-30):.3f}")
print(f"\n[maps] wrote {out}")
