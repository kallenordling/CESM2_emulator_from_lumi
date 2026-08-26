#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
assert_account
lumi_env_banner
# Launch a LARGE-ENSEMBLE eval (default 25 diffusion members per experiment).
#
# WHY 25
# ------
# The 5-member eval cannot resolve a precipitation bias. Measured at ep0490,
# ssp370, 10-yr mean anomaly: CESM2's own inter-member spread is 32% of the
# forced precipitation signal (temperature: 3.3%), so with 5-vs-5 members the
# t-statistic of the emulator-minus-CESM2 difference is |t| ~ 1.03 — the bias and
# the sampling noise are the same size, and NO grid point survives an
# FDR-controlled test. Temperature reaches |t| ~ 1.8 and 6-8% of area.
#
# Power scales through SE = sqrt(s_e^2/n_e + s_c^2/n_c), so the CESM2 term sets a
# FLOOR that emulator members cannot buy past. Held-out CESM2 members available:
# hist 10, ssp370 10, aaer 11, ghg 6 (the rest are in training and must not be
# used as reference). With n_c = 10:
#
#     n_emu x n_cesm    SE/s    |t|     note
#         5 x  5       0.632   1.03    current — 0% of precip area significant
#        25 x  5       0.490   1.33    wasted: CESM2 side left at 5
#         5 x 10       0.548   1.19    FREE (no compute, just use all held-out)
#        25 x 10       0.374   1.74    <-- this script
#        50 x 10       0.346   1.88    +8% power for +100% compute
#      1000 x 10       0.318   2.05    ceiling imposed by 10 CESM2 members
#
# 25 captures ~85% of the achievable gain; beyond that the CESM2 ensemble, not
# the emulator, is the binding constraint. Note the ceiling itself: even with
# infinite emulator members the TYPICAL grid point stays near p ~ 0.07, so expect
# precipitation to go from 0% to a modest, spatially concentrated percentage —
# not to light up.
#
# IMPORTANT: the emulator side is only half the job. Pass --n-ref-members 0 to
# scripts/paper_fig_maps.py so it uses ALL held-out CESM2 members instead of its
# default 5, otherwise n_c stays at 5 and most of this compute is wasted.
#
# COST
# ----
# ~5x the sampling of a 5-member eval. Sharding is per EXPERIMENT, so walltime is
# set by the heaviest single experiment (aaer/ghg, 251 years) on one GPU:
# ~6 h at 25 members. 12 h requested. Output NetCDF grows with member count —
# budget ~35-40 GB per eval dir on /scratch instead of ~10 GB.
#
# Usage (on LUMI, from the repo dir, after `git pull`):
#   bash submit_eval_ens25.sh <checkpoint.pt> [experiments] [output_subdir]
#
# Examples:
#   bash submit_eval_ens25.sh runs/run_mseyb_BCprect_490.pt "" ep0490_ens25
#   MEMBERS=50 bash submit_eval_ens25.sh runs/run_mseyb_BCprect_490.pt
#   # sampler test — more denoising steps, few members, hist only:
#   SAMPLE_STEPS=200 MEMBERS=10 bash submit_eval_ens25.sh \
#       runs/run_mseyb_BCprect_490.pt "hist" spec_steps200
#   bash submit_eval_ens25.sh runs/run_mseyb_BCprect_490.pt "hist ssp370" ep0490_ens25_ts
set -euo pipefail

CKPT="${1:?usage: bash submit_eval_ens25.sh <checkpoint.pt> [experiments] [output_subdir]}"
EXPERIMENTS="${2:-}"
SUBDIR="${3:-ens25_$(basename "${CKPT}" .pt)}"

MEMBERS="${MEMBERS:-25}"
# One rank per experiment: the shard unit is a whole experiment, so more ranks
# than experiments only leaves GPUs idle (see run_eval_aero.sh sharding notes).
NTASKS="${NTASKS:-6}"
WALLTIME="${WALLTIME:-12:00:00}"

REPO=${LUMI_REPO}
case "${CKPT}" in
    /*) ;;                       # already absolute
    *)  CKPT="${REPO}/${CKPT}" ;;
esac

OUTPUT_DIR="${LUMI_EVAL_OUT}/manual/${SUBDIR}"

EXPORTS="ALL,CHECKPOINT=${CKPT},OUTPUT_DIR=${OUTPUT_DIR},MEMBERS=${MEMBERS}"
[ -n "${EXPERIMENTS}" ] && EXPORTS="${EXPORTS},EXPERIMENTS=${EXPERIMENTS}"
# Passed explicitly rather than relying on --export=ALL, so the value that
# reaches the job is visible in the submit line and in the log banner. The
# sampler test needs this: the grid-scale variance errors (TREFHT 4.6x too
# much, PRECT 0.41x too little) look like a mis-set denoising schedule, and
# raising the step count is the way to find out — with no retraining.
[ -n "${SAMPLE_STEPS:-}" ] && EXPORTS="${EXPORTS},SAMPLE_STEPS=${SAMPLE_STEPS}"
[ -n "${BATCH_SIZE:-}" ]   && EXPORTS="${EXPORTS},BATCH_SIZE=${BATCH_SIZE}"

echo "[ens25] checkpoint  = ${CKPT}"
echo "[ens25] output_dir  = ${OUTPUT_DIR}"
echo "[ens25] members     = ${MEMBERS}   (vs 5 in the standard eval)"
echo "[ens25] experiments = ${EXPERIMENTS:-<all>}"
echo "[ens25] ntasks/gpus = ${NTASKS}    walltime = ${WALLTIME}"

if [ ! -f "${CKPT}" ]; then
    echo "[ens25] ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

# Sanity: a 25-member run into an existing 5-member dir would mix ensemble sizes
# across the per-experiment NetCDFs, and the member count is only recorded as a
# file attribute — an easy way to produce a silently inhomogeneous eval.
if [ -d "${OUTPUT_DIR}" ] && compgen -G "${OUTPUT_DIR}/*.nc" > /dev/null; then
    echo "[ens25] ERROR: ${OUTPUT_DIR} already holds NetCDF output." >&2
    echo "[ens25]        Use a fresh [output_subdir] or delete it first," >&2
    echo "[ens25]        otherwise experiments evaluated at different member" >&2
    echo "[ens25]        counts end up side by side in one directory." >&2
    exit 1
fi

# CLI options override the #SBATCH directives inside run_eval_aero.sh, so the
# runner stays a single source of truth instead of being copy-pasted per member
# count. --mem scales with the ensemble held in memory per rank.
sbatch --export="${EXPORTS}" \
       --job-name="eval_ens${MEMBERS}" \
       --ntasks="${NTASKS}" \
       --gpus-per-node="${NTASKS}" \
       --mem=256G \
       --time="${WALLTIME}" \
       run_eval_aero.sh

# The local sshfs mount that corresponds to LUMI_EVAL_PROJECT's scratch. Eval
# output follows LUMI_EVAL_PROJECT, not LUMI_PROJECT, so a hardcoded mount name
# points at the wrong project the moment the two differ (462001112 writes to
# mnt/lumi_sc, 462001328 to mnt/lumi_sc2).
case "${LUMI_EVAL_PROJECT}" in
    462001112) LOCAL_MOUNT="/home/nordling/mnt/lumi_sc"  ;;
    462001328) LOCAL_MOUNT="/home/nordling/mnt/lumi_sc2" ;;
    *)         LOCAL_MOUNT="<local mount of /scratch/project_${LUMI_EVAL_PROJECT}>" ;;
esac

cat <<EOF

[ens25] When it finishes, build the maps against ALL held-out CESM2 members
        (default is 5 — leaving it there discards most of this run's power):

  /home/nordling/miniconda3/envs/plotting/bin/python scripts/paper_fig_maps.py \\
      --eval-dir ${LOCAL_MOUNT}/eval_output/manual/${SUBDIR} \\
      --n-ref-members 0 \\
      --out plots/paper_fig_maps_ens${MEMBERS}.png \\
      --csv plots/map_stats_ens${MEMBERS}.csv

        (--n-ref-members 0 = use every held-out member: 10 hist / 10 ssp370 /
        11 aaer / 6 ghg. Panels will then differ in n_cesm, which the figure
        already reports per panel.)
EOF
