#!/bin/bash
# Assemble the Hugging Face Space: webapp/space/ plus the model code it imports.
#
#   bash webapp/build_space.sh [OUT_DIR] [--with-bundle]
#
# The model code is COPIED from the repo rather than duplicated in git, so the
# Space always runs the same UNetModel3D / ContinuousDDPM as training and eval.
# --with-bundle also copies webapp/bundle/ in (for a local run, or a Space that
# stores its weights itself instead of reading EMULATOR_BUNDLE_REPO).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${REPO}/webapp/_build/space}"
[[ "${OUT}" == --* ]] && OUT="${REPO}/webapp/_build/space"

rm -rf "${OUT}"
mkdir -p "${OUT}/models" "${OUT}/custom_diffusers"
cp "${REPO}"/webapp/space/{app.py,emulator.py,requirements.txt,README.md} "${OUT}/"
cp "${REPO}"/models/{video_net.py,rotary_embedding.py} "${OUT}/models/"
cp "${REPO}"/custom_diffusers/continuous_ddpm.py "${OUT}/custom_diffusers/"
touch "${OUT}/models/__init__.py" "${OUT}/custom_diffusers/__init__.py"

if [[ " $* " == *" --with-bundle "* ]]; then
    [[ -f "${REPO}/webapp/bundle/meta.json" ]] || { echo "no webapp/bundle — run export_bundle.py first" >&2; exit 1; }
    cp -r "${REPO}/webapp/bundle" "${OUT}/bundle"
fi
echo "Space assembled in ${OUT}"
