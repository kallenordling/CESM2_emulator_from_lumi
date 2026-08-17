#!/bin/bash
# Install this repo's extra python packages on Roihu, for the GPU (aarch64) side.
#
#     bash scripts/setup_roihu_pkgs.sh
#
# RUN THIS ON THE GPU LOGIN NODE (roihu-gpu-login2), not roihu-cpu.csc.fi.
# Roihu is heterogeneous: GPU nodes are Grace/ARM64, CPU partitions are x86_64.
# pip resolves wheels for the machine it runs on, so packages installed from the
# x86 login node are silently unusable on a GH200 job and vice versa. That is
# the same trap that produced a venv named diffesm-x86_64 full of ARM wheels.
#
# WHY --target AND NOT A VENV
# ---------------------------
# python-pytorch/2.10 is a singularity image. A venv built against a container
# python bakes in a symlink to that exact interpreter and breaks when the module
# is updated. A --target directory is just importable files on PYTHONPATH, which
# run_roihu.sh prepends automatically when the directory exists.
#
# WHAT IS DELIBERATELY NOT INSTALLED
# ----------------------------------
# torch/torchvision — the module provides them, built for GH200. Letting pip
# pull its own torch would replace a tuned CUDA build with a generic wheel, or
# fail outright on aarch64. --no-deps is NOT used (the extras have real deps),
# so torch is pinned out of the resolution by listing it as already satisfied:
# we install into --target with the module loaded, and pip sees the module's
# torch on the path. If pip still tries to fetch torch, add --no-deps and
# install the handful of leaf deps by hand rather than letting it win.
set -euo pipefail

export SITE=roihu
_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${_REPO_DIR}/lumi_env.sh"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "ERROR: this is $(uname -m). Roihu's GPU nodes are aarch64 and wheels" >&2
    echo "       installed here will not import in a gh200 job. ssh to the GPU" >&2
    echo "       login node (roihu-gpu-login2) and run this again." >&2
    echo "       For the CPU-side download env, use setup_env.sh instead." >&2
    exit 1
fi

module purge
eval "${SITE_MODULE_CMD}"

mkdir -p "${LUMI_PKGS}"
echo "[setup] arch    $(uname -m)"
echo "[setup] python  $(command -v python3) — $(python3 -V 2>&1)"
echo "[setup] target  ${LUMI_PKGS}"
python3 -c "import torch; print(f'[setup] torch   {torch.__version__} (from the module — not reinstalling)')"

# Keep this list in sync with requirements.txt, minus what the module ships
# (torch, numpy, scipy, matplotlib, pandas, scikit-learn, xarray, dask, netCDF4)
# and minus the plotting/attribution extras that training does not import.
PKGS=(
    einops
    einops-exts        # models/video_net.py: rearrange_many
    ema-pytorch
    beartype           # models/rotary_embedding.py
    diffusers
    accelerate
    hydra-core
    omegaconf
    huggingface_hub
)

python3 -m pip install --no-cache-dir --target="${LUMI_PKGS}" --upgrade "${PKGS[@]}"

export PYTHONPATH="${LUMI_PKGS}${PYTHONPATH:+:${PYTHONPATH}}"
echo
echo "[setup] verifying imports …"
fail=0
for m in einops einops_exts ema_pytorch beartype diffusers accelerate hydra omegaconf xarray torch; do
    if python3 -c "import ${m}" 2>/dev/null; then
        echo "  ok      ${m}"
    else
        echo "  MISSING ${m}"; fail=1
    fi
done
[ "${fail}" -eq 0 ] || { echo "[setup] some imports failed — see above" >&2; exit 1; }

echo
echo "[setup] done. run_roihu.sh picks ${LUMI_PKGS} up automatically."
echo "[setup] next:"
echo "  sbatch --account=${LUMI_ACCOUNT} --partition=gputest --time=00:15:00 \\"
echo "         run_roihu.sh scripts/smoke_test_model.py --sweep --batch 1"
