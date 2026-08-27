#!/bin/bash
# Install a Jupyter kernel that runs INSIDE the LUMI AI Factory container.
#
# WHY THIS EXISTS
# ---------------
# The LUMI web interface (www.lumi.csc.fi) starts a Jupyter server with a system
# Python. That Python puts ~/.local/lib/pythonX.Y/site-packages FIRST, so any
# pip --user install there shadows everything else. A broken or wrong-arch torch
# in ~/.local then fails late and cryptically, e.g.
#
#     OSError: ~/.local/lib/python3.11/site-packages/torch/lib/
#              libtorch_global_deps.so: cannot open shared object file
#
# — which is what a CUDA-build torch, truncated by an exhausted home quota,
# looks like on an AMD machine.
#
# Rather than fight the server's Python, this installs a KERNEL that launches
# ipykernel inside the same Singularity container the batch jobs use. The
# notebook then runs against the container's ROCm torch and the project's
# packages, whatever Python started the server.
#
# Usage (ON LUMI, once):
#     bash scripts/install_lumi_jupyter_kernel.sh
#     bash scripts/install_lumi_jupyter_kernel.sh --name mykernel
#
# Then in the LUMI web interface: start the Jupyter app as usual, open the
# notebook, and pick the kernel from the kernel menu (top right, or
# Kernel > Change Kernel).
set -euo pipefail

NAME="diffesm"
[ "${1:-}" = "--name" ] && NAME="${2:?--name needs a value}"

SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
VENV_SITE="${VENV_SITE:-/projappl/project_462001328/venvs/diffesm_laif/lib/python3.12/site-packages}"
EXTRA_PKGS="${EXTRA_PKGS:-/scratch/project_462001328/python_packages}"

KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/${NAME}"
WRAPPER="${KERNEL_DIR}/launch.sh"

if [ ! -f "${SIF}" ]; then
    echo "ERROR: container not found: ${SIF}" >&2
    echo "       Set SIF=/path/to/image.sif and re-run." >&2
    exit 1
fi

mkdir -p "${KERNEL_DIR}"

# ── the launcher ─────────────────────────────────────────────────────────────
# `module` is a shell FUNCTION, and a kernel is not started from a login shell,
# so it is usually undefined here — the same trap that silently killed jobs
# submitted over plain ssh. Source lmod's init if needed, and fall back to
# calling singularity by absolute path if the module system is unavailable.
cat > "${WRAPPER}" <<WRAP
#!/bin/bash
set -euo pipefail

# Ignore ~/.local on BOTH sides of the container boundary. This is the whole
# point: a broken user-site package must not shadow the container's packages.
export PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONPATH="${VENV_SITE}:${EXTRA_PKGS}"

if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
if command -v module >/dev/null 2>&1; then
    module use /appl/local/laifs/modules >/dev/null 2>&1 || true
    module load lumi-aif-singularity-bindings >/dev/null 2>&1 || true
fi

SINGULARITY_BIN="\$(command -v singularity || echo /usr/bin/singularity)"
exec "\${SINGULARITY_BIN}" exec "${SIF}" python -m ipykernel_launcher "\$@"
WRAP
chmod +x "${WRAPPER}"

# ── the kernel spec ──────────────────────────────────────────────────────────
cat > "${KERNEL_DIR}/kernel.json" <<SPEC
{
  "argv": ["${WRAPPER}", "-f", "{connection_file}"],
  "display_name": "DiffESM (LUMI container)",
  "language": "python"
}
SPEC

echo "installed kernel '${NAME}' -> ${KERNEL_DIR}"
echo
echo "container : ${SIF}"
echo "packages  : ${VENV_SITE}"
echo "            ${EXTRA_PKGS}"
echo
echo "Verifying it starts (this runs the container) ..."
if "${WRAPPER}" --version >/dev/null 2>&1 || true; then
    "$(command -v singularity || echo /usr/bin/singularity)" exec "${SIF}" \
        python -c "import torch, sys; print('  torch', torch.__version__);
print('  from ', torch.__file__);
print('  rocm/hip:', torch.version.hip)" 2>/dev/null \
        || echo "  (could not import torch inside the container — check SIF)"
fi
echo
echo "Next: start the Jupyter app in the LUMI web interface, open the notebook,"
echo "      and choose 'DiffESM (LUMI container)' from the kernel menu."
echo "      The preflight cell prints torch.__file__ — it must NOT be under"
echo "      ~/.local, or the wrong kernel is selected."
