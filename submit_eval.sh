#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
# Turnkey launcher for a MANUAL eval. Wraps `sbatch run_eval_aero.sh` with the
# correct `--export=ALL,CHECKPOINT=...,OUTPUT_DIR=...,EXPERIMENTS=...` so the env
# vars actually reach the job.
#
# WHY: a bare `VAR=val sbatch run_eval_aero.sh` prefix does NOT reliably
# propagate into the SLURM job on LUMI — that misfire once silently evaluated the
# wrong checkpoint (find_latest) into the default dir. This launcher removes that
# footgun: it builds one quoted --export string.
#
# Usage (on LUMI, from the repo dir, after `git pull`):
#   bash submit_eval.sh <checkpoint.pt> [experiments] [output_subdir]
#
#   <checkpoint.pt>  absolute path, or a path relative to the repo (e.g.
#                    runs/run_mseyb_852.pt — resolved under the projappl repo).
#   [experiments]    space-separated scenario names for --experiments (e.g.
#                    "ssp126" or "hist ssp370 ssp126"); empty/omit = all 5.
#   [output_subdir]  name under /scratch/.../eval_output/manual/ (default:
#                    manual_<ckpt-basename>).
#
# Examples:
#   bash submit_eval.sh runs/run_mseyb_852.pt ssp126 ep0852_pcafix
#   bash submit_eval.sh runs/run_sensfix_955.pt          # all scenarios
set -euo pipefail

CKPT="${1:?usage: bash submit_eval.sh <checkpoint.pt> [experiments] [output_subdir]}"
EXPERIMENTS="${2:-}"
SUBDIR="${3:-manual_$(basename "${CKPT}" .pt)}"

REPO=${LUMI_REPO}
case "${CKPT}" in
    /*) ;;                       # already absolute
    *)  CKPT="${REPO}/${CKPT}" ;;
esac

OUTPUT_DIR="${LUMI_EVAL_OUT}/manual/${SUBDIR}"

EXPORTS="ALL,CHECKPOINT=${CKPT},OUTPUT_DIR=${OUTPUT_DIR}"
[ -n "${EXPERIMENTS}" ] && EXPORTS="${EXPORTS},EXPERIMENTS=${EXPERIMENTS}"

echo "[submit_eval] checkpoint  = ${CKPT}"
echo "[submit_eval] output_dir  = ${OUTPUT_DIR}"
echo "[submit_eval] experiments = ${EXPERIMENTS:-<all 5>}"

if [ ! -f "${CKPT}" ]; then
    echo "[submit_eval] ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

# Quote the whole --export so a multi-word EXPERIMENTS ("hist ssp370 ssp126")
# survives as one value.
sbatch --export="${EXPORTS}" run_eval_aero.sh
