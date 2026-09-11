#!/usr/bin/env python3
"""
================================================================================
 DUMP THE CONDITIONING EOFs FROM A CHECKPOINT
================================================================================

Run it on LUMI, where the checkpoints live:

    bash run_dump_eofs.sh /path/to/run_mseyb_BCprect_878.pt

WHY
---
The conditioning channels are PCA-denoised before the model ever sees them —
`n_components_cond: [30, 5, 5]` for [CO2, SUL, BC]. The maps handed to the
network are reconstructions from those components, so the model's whole
vocabulary of aerosol patterns is FIVE spatial modes per species. Anything the
emulator can say about "which emissions, and where" is a statement about those
modes; a per-grid-cell answer would be extrapolation off the manifold it was
trained on.

This pulls those modes out of a checkpoint so they can be plotted and named.

WHAT IT WRITES
--------------
One small .npz (a few MB against the checkpoint's ~780 MB), holding for every
(basis, channel):

    components_<basis>_<channel>       (n_components, H, W)  the EOF maps
    variance_<basis>_<channel>         (n_components,)       explained variance ratio
    mean_<basis>_<channel>             (H, W)                the basis mean

`basis` is "reference" for ckpt["PCA"]["cond"], and the scenario name for each
entry of ckpt["PCA"]["per_scenario"]. THE PER-SCENARIO BASES ARE THE ONES EVAL
ACTUALLY USES for trained scenarios — the reference basis is the fallback, and
an unseen scenario gets a fresh fit instead (which is the open windowed/fresh-PCA
confound in the RAMIP comparison). Both are dumped so the two can be compared.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not refit anything. Refitting on the cond files would give a basis that
is *nearly* the checkpoint's and not identical, and the difference is exactly
the confound above.
"""

import argparse
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", help="path to a .pt checkpoint")
parser.add_argument("--out", default="cond_eofs.npz", help="output .npz")
parser.add_argument("--shape", default="192,288",
                    help="lat,lon grid the flattened EOFs unfold to")
args = parser.parse_args()

height, width = (int(v) for v in args.shape.split(","))

# weights_only=False: the payload contains pickled sklearn PCA objects, which is
# the whole point of reading it. Only ever run this on your own checkpoints.
print(f"[dump] loading {args.checkpoint} (CPU, this takes a minute)", flush=True)
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

pca_state = checkpoint.get("PCA")
if pca_state is None:
    sys.exit("[error] this checkpoint has no 'PCA' entry — it predates PCA "
             "persistence, and eval against it would skip PCA entirely")

# Channel names in the order config_data lists cond_vars. Only used for labels;
# the count comes from the checkpoint.
CHANNEL_NAMES = ["CO2", "SUL", "BC"]

bases = {"reference": pca_state.get("cond")}
for name, entry in (pca_state.get("per_scenario") or {}).items():
    bases[name] = entry.get("cond")

payload = {}
for basis_name, channel_list in bases.items():
    if not channel_list:
        print(f"[dump] {basis_name}: no cond basis, skipped")
        continue
    for index, pca in enumerate(channel_list):
        if pca is None:
            continue
        channel = (CHANNEL_NAMES[index] if index < len(CHANNEL_NAMES)
                   else f"ch{index}")
        components = np.asarray(pca.components_)
        if components.shape[1] != height * width:
            sys.exit(f"[error] {basis_name}/{channel}: {components.shape[1]} "
                     f"features do not unfold to {height}x{width} — pass the "
                     f"right --shape")
        payload[f"components_{basis_name}_{channel}"] = components.reshape(
            -1, height, width).astype(np.float32)
        payload[f"variance_{basis_name}_{channel}"] = np.asarray(
            pca.explained_variance_ratio_, dtype=np.float32)
        payload[f"mean_{basis_name}_{channel}"] = np.asarray(
            pca.mean_, dtype=np.float32).reshape(height, width)
        print(f"[dump] {basis_name:10s} {channel:4s} "
              f"{components.shape[0]} components, "
              f"variance explained "
              f"{100 * float(pca.explained_variance_ratio_.sum()):.1f}%")

if not payload:
    sys.exit("[error] nothing dumped — the checkpoint's PCA entry held no "
             "cond bases")

np.savez_compressed(args.out, **payload)
print(f"[dump] wrote {args.out}")
