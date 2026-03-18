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
echo "Step 1: module --force purge"
module --force purge
echo "Step 2: module use /appl/local/laifs/modules"
module use /appl/local/laifs/modules
echo "Step 3: module load lumi-aif-singularity-bindings"
module load lumi-aif-singularity-bindings
echo "Step 4: finding SIF container"

SIF=$(ls /appl/local/laifs/containers/lumi-multitorch-full-*.sif 2>/dev/null | sort -V | tail -1)
if [[ -z "${SIF}" ]]; then
    echo "ERROR: no lumi-multitorch-full container found in /appl/local/laifs/containers/"
    echo "Available containers:"
    ls /appl/local/laifs/containers/ 2>/dev/null || echo "  (directory not found)"
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
