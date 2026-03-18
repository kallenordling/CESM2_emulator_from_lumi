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

SIF=$(ls /appl/local/laifs/containers/lumi-multitorch-full-*.sif 2>/dev/null | sort -V | tail -1)
if [[ -z "${SIF}" ]]; then
    echo "ERROR: no lumi-multitorch-full container found in /appl/local/laifs/containers/"
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

echo "Installing packages from requirements.txt ..."
singularity exec "${SIF}" bash -c "
    source ${VENV_DIR}/bin/activate
    pip install --upgrade pip
    pip install -r ${SCRIPT_DIR}/requirements.txt
"

echo ""
echo "=== Done ==="
echo ""
echo "Now update VENV in run2_aero.sh:"
echo "  VENV=${VENV_DIR}/bin/activate"
