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
# --no-deps IS MANDATORY HERE. THIS IS THE WHOLE PROBLEM.
# ------------------------------------------------------
# `pip install --target` does NOT see what is already importable. It treats the
# target directory as an empty world, so it re-resolves every dependency from
# scratch even when the module already provides it. The first version of this
# script omitted --no-deps and pip duly installed torch 2.13.0, numpy 2.5.2 and
# ~3 GB of nvidia-* CUDA wheels into the target. Because run_roihu.sh prepends
# the target to PYTHONPATH, those SHADOW the module's torch 2.10.0+cu130 — the
# build actually tuned for GH200 — and the module's numpy. The visible symptom
# was `xarray` failing to import: the module's xarray was being loaded against
# the shadowing numpy.
#
# So: install with --no-deps, then resolve the genuinely-missing leaves one at
# a time, refusing ever to install anything in BLOCK. A package that pip says
# is "not installed" is usually already present via the module.
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

# Start clean. A target polluted by a previous run without --no-deps contains a torch
# that shadows the module's, and no amount of reinstalling on top removes it.
if [ -d "${LUMI_PKGS}" ]; then
    echo "[setup] removing the existing target ${LUMI_PKGS}"
    du -sh "${LUMI_PKGS}" 2>/dev/null | sed 's/^/[setup]   was /'
    rm -rf "${LUMI_PKGS}"
fi

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

# Never let these into the target: the module owns them, and a shadowing
# copy is worse than useless. torch especially — the module's build is the
# CUDA/GH200 one and a pip wheel silently replaces it.
BLOCK="torch torchvision torchaudio numpy scipy pandas matplotlib xarray dask netCDF4 nvidia-* triton cuda-* sympy"

pip_add() {   # install ONE package, never a blocked one, never its deps
    local p="$1"
    for b in ${BLOCK}; do
        # shellcheck disable=SC2254
        case "${p}" in ${b}) echo "  [blocked] ${p} — provided by the module"; return 0 ;; esac
    done
    python3 -m pip install --no-cache-dir --no-deps --target="${LUMI_PKGS}" \
        --upgrade "${p}" >/dev/null 2>&1 \
        && echo "  installed ${p}" || { echo "  FAILED    ${p}" >&2; return 1; }
}

echo "[setup] installing (--no-deps) …"
for p in "${PKGS[@]}"; do pip_add "${p}"; done

export PYTHONPATH="${LUMI_PKGS}${PYTHONPATH:+:${PYTHONPATH}}"

# Resolve the leaves --no-deps left out, by asking python what is actually
# missing rather than guessing. Import name != package name for several, hence
# the map. Loops because installing one dep can reveal the next.
declare -A PKG_OF=(
    [antlr4]=antlr4-python3-runtime  [PIL]=Pillow  [yaml]=PyYAML
    [importlib_metadata]=importlib-metadata  [hf_xet]=hf-xet
    [huggingface_hub]=huggingface_hub  [safetensors]=safetensors
    [regex]=regex  [filelock]=filelock  [packaging]=packaging
    [psutil]=psutil  [tqdm]=tqdm  [requests]=requests  [fsspec]=fsspec
    [typing_extensions]=typing_extensions
)
CHECK="einops einops_exts ema_pytorch beartype diffusers accelerate hydra omegaconf xarray numpy torch"

echo
echo "[setup] verifying imports …"
for round in 1 2 3 4; do
    missing_mod=""
    for m in ${CHECK}; do
        err="$(python3 -c "import ${m}" 2>&1)" && continue
        # "No module named 'X'" names the DEPENDENCY, which may not be m itself.
        dep="$(printf '%s' "${err}" | sed -n "s/.*No module named '\([A-Za-z0-9_]*\)'.*/\1/p" | tail -1)"
        [ -n "${dep}" ] && missing_mod="${missing_mod} ${dep}"
    done
    missing_mod="$(printf '%s\n' ${missing_mod} | sort -u | tr '\n' ' ')"
    [ -z "${missing_mod// /}" ] && break
    echo "[setup] round ${round}: resolving${missing_mod}"
    for m in ${missing_mod}; do pip_add "${PKG_OF[${m}]:-${m}}" || true; done
done

fail=0
for m in ${CHECK}; do
    if out="$(python3 -c "import ${m}, sys; print(getattr(${m},'__version__','?'))" 2>&1)"; then
        printf '  ok      %-16s %s\n' "${m}" "${out}"
    else
        printf '  MISSING %-16s %s\n' "${m}" "$(printf '%s' "${out}" | tail -1)"; fail=1
    fi
done

# The point of the whole exercise: torch must still be the MODULE's build.
python3 - <<'PYCHK' || fail=1
import torch, os, sys
p = os.path.dirname(torch.__file__)
tgt = os.environ.get("LUMI_PKGS", "")
print(f"  torch   {torch.__version__}  from {p}")
if tgt and p.startswith(tgt):
    print("  ERROR: torch is being loaded from the pip target, SHADOWING the",
          "module's GH200 build. Delete it:", file=sys.stderr)
    print(f"           rm -rf {tgt}/torch* {tgt}/nvidia* {tgt}/triton*", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("  note: no CUDA visible — expected on a login node, checked again in the job")
PYCHK

[ "${fail}" -eq 0 ] || { echo "[setup] some imports failed — see above" >&2; exit 1; }

echo
echo "[setup] done. run_roihu.sh picks ${LUMI_PKGS} up automatically."
echo "[setup] next:"
echo "  sbatch --account=${LUMI_ACCOUNT} --partition=gputest --time=00:15:00 \\"
echo "         run_roihu.sh scripts/smoke_test_model.py --sweep --batch 1"
