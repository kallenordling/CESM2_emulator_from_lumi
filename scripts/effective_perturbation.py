#!/usr/bin/env python3
"""
================================================================================
 WHAT PERTURBATION DOES THE MODEL ACTUALLY RECEIVE?
================================================================================

    python scripts/effective_perturbation.py <ckpt> --base <a.nc> --perturbed <b.nc> \
        --species SUL --basis ssp370 --region 20,50,100,145

Runs BOTH cond files through the real pipeline — COND_NORM clip ranges,
`normalize`, `smooth_cond_spatial`, then the persisted PCA basis via
`apply_pca_denoise` — and reports the difference the network would see.

This is the question that decides whether a regional attribution is possible at
all. A sharp source-region box is high spatial frequency; the pipeline blurs it
(sigma=2 gaussian on the aerosols) and then projects it onto five smooth global
EOFs. If almost nothing survives, or if what survives lands somewhere else, then
"which region drives N" cannot be answered by perturbing that region — the
model cannot represent the question.
"""

import argparse
import os
import sys

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.climate_dataset import (normalize, smooth_cond_spatial,
                                  set_minmax_override, apply_pca_denoise)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
parser.add_argument("--base", required=True)
parser.add_argument("--perturbed", required=True)
parser.add_argument("--species", required=True)
parser.add_argument("--basis", default="ssp370")
parser.add_argument("--region", required=True,
                    help="lat_min,lat_max,lon_min,lon_max of the box perturbed")
parser.add_argument("--cond-vars", default="CO2,SUL,BC")
parser.add_argument("--smooth-sigma", default="0,2,2")
parser.add_argument("--save-maps", metavar="NPZ",
                    help="write the requested and effective change maps so the "
                         "result can be plotted rather than only tabulated")
args = parser.parse_args()

cond_vars = args.cond_vars.split(",")
sigmas = [float(s) for s in args.smooth_sigma.split(",")]
channel = cond_vars.index(args.species)
lat_min, lat_max, lon_min, lon_max = (float(v) for v in args.region.split(","))

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cond_norm = checkpoint.get("COND_NORM")
if not cond_norm:
    sys.exit("[error] checkpoint has no COND_NORM")
set_minmax_override(cond_norm)
basis = (checkpoint.get("PCA", {}).get("per_scenario", {})
         .get(args.basis, {}).get("cond"))
if basis is None:
    sys.exit(f"[error] no persisted '{args.basis}' basis")

def pipeline(path):
    """normalize -> stack -> smooth -> PCA-reconstruct, as training does."""
    raw = xr.open_dataset(path)
    time_dim = "time"
    if time_dim not in raw.dims and "year" in raw.dims:
        raw = raw.rename({"year": time_dim})
    raw = raw[cond_vars]
    norm = raw.map(normalize)
    stacked = norm.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"])
    stacked = stacked.transpose("var", time_dim, "lat", "lon")
    field = stacked.values.astype(np.float32)
    lat = norm["lat"].values
    lon = norm["lon"].values
    raw.close()
    smoothed = smooth_cond_spatial(field, sigmas, "gaussian", cond_vars)
    reconstructed = apply_pca_denoise(smoothed[channel], basis[channel])
    return smoothed[channel], reconstructed, lat, lon

pre_base, post_base, lat, lon = pipeline(args.base)
pre_pert, post_pert, _, _ = pipeline(args.perturbed)

lon_grid, lat_grid = np.meshgrid(lon, lat)
inside = ((lat_grid >= lat_min) & (lat_grid <= lat_max)
          & (lon_grid >= lon_min) & (lon_grid <= lon_max))
weights = np.cos(np.deg2rad(lat))[:, None]
area = np.broadcast_to(weights, inside.shape)

# The final year, where the ramp is fully applied.
delta_requested = (pre_pert - pre_base)[-1]      # after smoothing, before PCA
delta_effective = (post_pert - post_base)[-1]    # what the network receives

def inside_share(d):
    total = float(np.average(np.abs(d), weights=area))
    within = float(np.average(np.abs(d) * inside, weights=area))
    return 100 * within / total if total > 0 else float("nan")

var_req = float(delta_requested.var())
var_eff = float(delta_effective.var())
print(f"[eff] species {args.species}, basis {args.basis}, "
      f"box lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}")
print(f"[eff] requested change (post-smoothing) : var {var_req:.4g}, "
      f"{inside_share(delta_requested):.1f}% of |change| inside the box")
print(f"[eff] effective change (post-PCA)       : var {var_eff:.4g}, "
      f"{inside_share(delta_effective):.1f}% of |change| inside the box")
print(f"[eff] variance surviving the projection : "
      f"{100 * var_eff / var_req:.2f}%" if var_req > 0 else "n/a")
if var_req > 0 and var_eff > 0:
    r = np.corrcoef(delta_requested.ravel(), delta_effective.ravel())[0, 1]
    print(f"[eff] shape correlation requested vs effective : {r:.3f}")

if args.save_maps:
    np.savez_compressed(
        args.save_maps,
        requested=delta_requested.astype(np.float32),
        effective=delta_effective.astype(np.float32),
        lat=lat, lon=lon, inside=inside,
        stats=np.array([var_req, var_eff, inside_share(delta_requested),
                        inside_share(delta_effective),
                        float(np.corrcoef(delta_requested.ravel(),
                                          delta_effective.ravel())[0, 1])]),
        meta=np.array([args.species, args.basis, args.region], dtype=object),
        allow_pickle=True)
    print(f"[eff] wrote {args.save_maps}")
