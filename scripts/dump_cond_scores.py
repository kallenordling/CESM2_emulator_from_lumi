#!/usr/bin/env python3
"""
================================================================================
 PC TIME SERIES FOR THE CONDITIONING MODES
================================================================================

Run it on LUMI:

    bash run_dump_scores.sh runs/run_mseyb_BCprect_863.pt cond_scores_ep0863.npz

WHY
---
scripts/dump_cond_eofs.py gives the EOF maps — WHERE each mode puts its weight.
An EOF alone cannot say WHEN a mode acts or HOW MUCH of the emission change it
accounts for in a given decade. That is the score time series, and it is not in
the checkpoint: it has to be recomputed by projecting the conditioning data onto
the persisted basis.

THE PREPROCESSING MUST MATCH EXACTLY
------------------------------------
The cond pipeline is clip -> normalize to [-1,1] -> spatial smooth -> PCA, and
the clip ranges are per-channel percentiles PERSISTED in the checkpoint as
COND_NORM. Getting any of that subtly wrong produces scores that look plausible
and are wrong, so nothing here is reimplemented: it calls the project's own
`normalize`, `smooth_cond_spatial` and `set_minmax_override`, in the order
eval_aero.build_cond_tensor uses them.

AND IT IS CHECKED
-----------------
sklearn stores `explained_variance_[k]`, which IS the variance of score k on the
data the basis was fitted on. So projecting the same cond file must reproduce
it. The script asserts that agreement and REFUSES to write a file that fails —
the replication is verified, not assumed.

WHAT IT WRITES
--------------
    scores_<channel>      (T, n_components)   PC time series
    variance_<channel>    (n_components,)     explained_variance_ (not the ratio)
    ratio_<channel>       (n_components,)     explained_variance_ratio_
    years                 (T,)
"""

import argparse
import os
import sys

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.climate_dataset import (normalize, smooth_cond_spatial,
                                  set_minmax_override)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
parser.add_argument("--cond-file", required=True,
                    help="the cond file the basis was fitted on")
parser.add_argument("--basis", default="ssp370",
                    help="which per-scenario basis to project onto")
parser.add_argument("--out", default="cond_scores.npz")
parser.add_argument("--cond-vars", default="CO2,SUL,BC")
parser.add_argument("--smooth-sigma", default="0,2,2")
parser.add_argument("--tol", type=float, default=0.05,
                    help="max relative disagreement with explained_variance_")
args = parser.parse_args()

cond_vars = args.cond_vars.split(",")
sigmas = [float(s) for s in args.smooth_sigma.split(",")]

print(f"[scores] loading {args.checkpoint}", flush=True)
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

# The clip ranges FIRST — normalize() reads them through the module-level
# override, so this must happen before any field is normalised.
cond_norm = checkpoint.get("COND_NORM")
if not cond_norm:
    sys.exit("[error] checkpoint has no COND_NORM; the clip ranges would be "
             "recomputed from this file instead of matching training")
set_minmax_override(cond_norm)
print(f"[scores] COND_NORM: " + ", ".join(f"{k}={v}" for k, v in cond_norm.items()))

pca_state = checkpoint.get("PCA") or {}
basis = (pca_state.get("per_scenario") or {}).get(args.basis, {}).get("cond")
if basis is None:
    sys.exit(f"[error] no persisted '{args.basis}' cond basis in this checkpoint")

# ── the same load/normalise/stack/smooth sequence as build_cond_tensor ───────
raw = xr.open_dataset(args.cond_file)
time_dim = "time"
if time_dim not in raw.dims and "year" in raw.dims:
    raw = raw.rename({"year": time_dim})
raw = raw[cond_vars]
norm = raw.map(normalize)
stacked = norm.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"])
stacked = stacked.transpose("var", time_dim, "lat", "lon")
field = stacked.values.astype(np.float32)                    # (n_vars, T, H, W)
years = np.asarray(raw[time_dim].values).astype(int)
raw.close()

field = smooth_cond_spatial(field, sigmas, "gaussian", cond_vars)
print(f"[scores] cond field {field.shape}, years {years.min()}-{years.max()}")

payload = {"years": years}
failures = []
for index, channel in enumerate(cond_vars):
    pca = basis[index]
    flat = field[index].reshape(len(years), -1).astype(np.float64)
    scores = pca.transform(flat)                             # (T, n_components)

    measured = scores.var(axis=0, ddof=1)
    stored = np.asarray(pca.explained_variance_)
    # Compare only where the stored variance is meaningful: a degenerate channel
    # (constant in time) has stored variance 0 and NaN ratios, and there is
    # nothing to verify.
    ratio = np.asarray(pca.explained_variance_ratio_)
    # JUDGE ONLY THE MODES THAT MATTER. Relative error on a component holding
    # 1e-6 of the variance is dominated by float noise and says nothing about
    # whether the preprocessing matches; the leading modes are the test.
    usable = np.isfinite(stored) & (stored > 1e-12) & np.isfinite(ratio) & (ratio > 1e-2)
    print(f"[scores] {channel:4s} {scores.shape}")
    for k in range(min(5, len(stored))):
        rel = (abs(measured[k] - stored[k]) / stored[k]
               if np.isfinite(stored[k]) and stored[k] > 1e-12 else float("nan"))
        print(f"           EOF{k+1}: stored {stored[k]:.6g}  measured "
              f"{measured[k]:.6g}  rel {rel:.2e}"
              + ("   <- judged" if usable[k] else "   (ignored, <1% var)"))
    if usable.any():
        relative = np.abs(measured[usable] - stored[usable]) / stored[usable]
        worst = float(relative.max())
        ok = worst <= args.tol
        print(f"[scores] {channel:4s} worst over judged modes = {worst:.2e}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{channel}: {worst:.2e}")
    else:
        print(f"[scores] {channel:4s} {scores.shape} — DEGENERATE basis "
              f"(zero stored variance), nothing to verify")

    payload[f"scores_{channel}"] = scores.astype(np.float32)
    payload[f"variance_{channel}"] = stored.astype(np.float64)
    payload[f"ratio_{channel}"] = np.asarray(
        pca.explained_variance_ratio_, dtype=np.float64)

if failures:
    sys.exit("[error] the projection does NOT reproduce the stored variances "
             f"({'; '.join(failures)}) — the preprocessing does not match "
             "training, so these scores would be wrong. Not writing a file.")

np.savez_compressed(args.out, **payload)
print(f"[scores] verified against explained_variance_ — wrote {args.out}")
