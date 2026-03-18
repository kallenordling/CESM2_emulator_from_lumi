#!/bin/bash
# setup_venv_laif.sh — one-time setup to create a venv inside the LAIF container.
#
# Run this ONCE on a LUMI login node (or interactive job) after switching to
# the LUMI AI Factory containers.
#
# Usage:
#   bash setup_venv_laif.sh [project_id]
#
# Example:
#   bash setup_venv_laif.sh project_462001328
#
# After this script finishes, update VENV in run2_aero.sh to:
#   VENV=/projappl/<project_id>/venvs/diffesm_laif/bin/activate

set -euo pipefail

# 'module' is a shell function initialised only in login shells.
# Re-exec as a login shell if it's not available yet.
if ! type module &>/dev/null 2>&1; then
    exec bash -l "$0" "$@"
fi

PROJECT="${1:-project_462001328}"
VENV_DIR="/projappl/${PROJECT}/venvs/diffesm_laif"

echo "=== LUMI AI Factory venv setup ==="
echo "Project : ${PROJECT}"
echo "Venv    : ${VENV_DIR}"
echo ""

# ── Load LAIF modules ─────────────────────────────────────────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: container not found at ${SIF}"
    echo "Available containers:"
    ls /appl/local/laifs/containers/*.sif 2>/dev/null || echo "  (none found)"
    exit 1
fi
echo "Container: ${SIF}"
echo ""

# ── Create venv with system-site-packages ────────────────────────────────────
# --system-site-packages makes torch, numpy, etc. available from the container
# without reinstalling them (saves space and avoids version conflicts).
echo "Creating venv at ${VENV_DIR} ..."
singularity exec "${SIF}" python -m venv --system-site-packages "${VENV_DIR}"

# ── Install project-specific packages ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve to real path in case /project/ is a symlink not visible inside
# the container (LUMI mounts /pfs/lustrep1/... but not always /project/).
REAL_SCRIPT_DIR="$(realpath "${SCRIPT_DIR}" 2>/dev/null || echo "${SCRIPT_DIR}")"
REQ_FILE="${REAL_SCRIPT_DIR}/requirements.txt"

echo "Installing packages from ${REQ_FILE} ..."
singularity exec "${SIF}" bash -c "
    source ${VENV_DIR}/bin/activate
    pip install --upgrade pip
    pip install -r ${REQ_FILE}
"

echo ""
echo "=== Done ==="
echo ""
echo "Now update VENV in run2_aero.sh:"
echo "  VENV=${VENV_DIR}/bin/activate"
