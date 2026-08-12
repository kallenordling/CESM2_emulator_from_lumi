#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
# CPU-only: download raw monthly CESM2 source data (LENS2 + SF AAER/GHG) from
# OSDF, inside the LAIF singularity container (project venv injected). Wraps
# download_lens2.py (d651056) + download_cesm_lens_sf.py (d651055).
#
# The download scripts SKIP files that already exist, so this is safe to re-run
# and only fetches what's missing. They do NOT exit non-zero on failed files
# (transient OSDF errors silently left 268 PRECT gaps last time), so this
# wrapper (a) retries up to PASSES times and (b) VERIFIES the downloaded tree
# against the complete TREFHT tree and fails loudly if anything is missing.
#
# Writes to:
#   lens2/LENS2/<VAR>/LE2-*/   and   sf/{AAER,GHG}/<VAR>/
#
# Usage on LUMI (from the repo dir, after `git pull`):
#   bash run_download_data.sh [VARIABLE] [PASSES]   # default PRECT, 3 passes
# e.g.
#   bash run_download_data.sh PRECT
#
# Long downloads; run inside a batch/interactive alloc, NOT on the login node:
#   srun --account=${LUMI_ACCOUNT} --partition=small --time=06:00:00 \
#        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
#        bash run_download_data.sh PRECT
set -euo pipefail

VARIABLE="${1:-PRECT}"
PASSES="${2:-3}"

DATA_ROOT=${LUMI_DATA}
LENS_DIR="${DATA_ROOT}/lens2"
SF_DIR="${DATA_ROOT}/sf"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_prepare_data.sh).
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[download] VARIABLE=${VARIABLE}  PASSES=${PASSES}"
echo "[download] SIF=${SIF}"

for pass in $(seq 1 "${PASSES}"); do
    echo "[download] === pass ${pass}/${PASSES} ==="

    echo "[download] LENS2 → ${LENS_DIR}/LENS2/${VARIABLE}/ …"
    singularity exec "${SIF}" python download_lens2.py \
        --variable "${VARIABLE}" \
        --output   "${LENS_DIR}" \
        --workers  4

    echo "[download] SF (AAER,GHG) → ${SF_DIR}/{AAER,GHG}/${VARIABLE}/ …"
    singularity exec "${SIF}" python download_cesm_lens_sf.py \
        --ensemble AAER GHG \
        --variable "${VARIABLE}" \
        --output   "${SF_DIR}" \
        --workers  4
done

# Verify against the TREFHT tree (known complete): every TREFHT file must have
# a same-named ${VARIABLE} counterpart. Exits 1 + lists missing files if not.
echo "[download] verifying ${VARIABLE} tree against TREFHT …"
singularity exec "${SIF}" python - "${VARIABLE}" "${LENS_DIR}" "${SF_DIR}" <<'EOF'
import sys
from pathlib import Path

var, lens_dir, sf_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

def diff_tree(trefht_dir, var_dir):
    ref = {f.name for f in trefht_dir.iterdir() if f.name.endswith(".nc")}
    got = {f.name for f in var_dir.iterdir() if f.name.endswith(".nc")} if var_dir.exists() else set()
    want = {n.replace(".TREFHT.", f".{var}.") for n in ref if ".TREFHT." in n}
    return sorted(want - got)

missing = []
for mdir in sorted((lens_dir / "LENS2" / "TREFHT").iterdir()):
    if mdir.is_dir():
        missing += diff_tree(mdir, lens_dir / "LENS2" / var / mdir.name)
for ens in ("AAER", "GHG"):
    missing += diff_tree(sf_dir / ens / "TREFHT", sf_dir / ens / var)

if missing:
    print(f"VERIFY FAILED: {len(missing)} {var} files still missing:")
    for m in missing:
        print(f"  {m}")
    sys.exit(1)
print(f"VERIFY OK: {var} tree complete (matches TREFHT).")
EOF

echo "[download] done — ${VARIABLE} complete. Next: bash run_prepare_data.sh ${VARIABLE}"
