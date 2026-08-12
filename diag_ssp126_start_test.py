#!/usr/bin/env python3
"""diag_ssp126_start_test.py
============================
Decisively answer WHY the emulator starts ssp126 ~1.5-2 C colder than ssp370 at
2015, by reusing the REAL eval pipeline so it diffs exactly what the model
consumes (zero reimplementation of cond-building or sampling).

Background (established by prior architect+engineer work):
  * At inference the model is MEMORYLESS across years — generate_timeseries
    (eval_aero.py:344-349) denoises each year as an independent 1-frame clip
    (years on the batch axis, temporal axis f=1). So identical 2015 cond MUST
    give identical 2015 output.
  * ssp126 nonetheless starts 1.5-2 C colder (below the 1850-1900 baseline =
    physically impossible). Either the CONSUMED 2015 cond actually differs, or
    the cold start came from an old setup. This script settles it.
  * Eval applies per-scenario PCA-basis routing: ssp126 (OOD, no own basis) ->
    the reference basis (aaer); ssp370 -> its own basis (eval_aero.py:1914-1921).
    PCA may be ABSENT in mount checkpoints, in which case eval skips PCA — both
    cases are handled here exactly as eval handles them.

What is reused FROM eval_aero (by import, no copy):
  * EXPERIMENTS            — cond_file / time_dim per scenario
  * build_cond_tensor      — the EXACT tensor the model consumes
  * load_model             — model + pca_state from a checkpoint
  * generate_timeseries    — the EXACT sampling path eval uses
  * extract_years          — robust cftime/int year extraction
  * COND_VARS, TARGET_VAR  — channel / target names

What is REPLICATED (only a few arg-resolution lines, NOT logic) and why:
  * N_COMP_COND / COND_SMOOTH_SIGMA / COND_SMOOTH_METHOD are read from
    configs/config_data.yaml inside eval_aero.main() (lines 1824-1839), not
    exported as module constants. We replicate those ~6 lines verbatim so the
    args handed to build_cond_tensor are byte-identical to eval's.
  * The per-scenario -> reference PCA-basis selection (eval lines 1811-1921) also
    lives inside main(). We replicate it in resolve_pca_basis() exactly: pull
    pca_state["cond"] as the reference, pca_state["per_scenario"][name]["cond"]
    if present, gated on N_COMP_COND being non-None — identical branch logic.
  * scheduler = instantiate(cfg.scheduler) from configs/config_aero.yaml — one
    line, same as eval line 1840.

generate_timeseries on a length-1 cond: cond_tensor is (C, T=1, H, W). The batch
loop (eval_aero.py:344) runs once with B=1, building gen = randn(1,1,1,H,W) and
denoising a single (1,C,1,H,W) frame — exactly one independent year. Output is
(T=1, H, W) in normalised space; eval denorms with *21.0 + 4.5 (line 1955), which
we mirror.

Usage:
  TEST A only (no GPU needed — pure cond diff):
      python diag_ssp126_start_test.py --test-a-only
  Full A+B (needs 1 GPU for the forward pass):
      python diag_ssp126_start_test.py --checkpoint /projappl/.../runs/run_mseyb_852.pt

See run_ssp126_start_test.sh for the LUMI container invocation.
"""

import lumi_paths as L
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

# ── reuse the REAL eval pipeline (no reimplementation) ──────────────────────────
import eval_aero
from eval_aero import (
    EXPERIMENTS,
    COND_VARS,
    build_cond_tensor,
    load_model,
    generate_timeseries,
    extract_years,
)

CONFIG_AERO = "configs/config_aero.yaml"
CONFIG_DATA = "configs/config_data.yaml"
DEFAULT_CKPT = (
    f"{L.REPO}/runs/run_mseyb_852.pt"
)


def get_experiment(name: str) -> dict:
    for e in EXPERIMENTS:
        if e["name"] == name:
            return e
    raise KeyError(f"experiment '{name}' not in EXPERIMENTS "
                   f"({[e['name'] for e in EXPERIMENTS]})")


def resolve_cond_args():
    """Replicate eval_aero.main() lines 1824-1839 verbatim.

    Returns (N_COMP_COND_or_None_if_no_pca_marker, COND_SMOOTH_SIGMA,
    COND_SMOOTH_METHOD).  N_COMP_COND is finalised per-checkpoint in
    resolve_pca_basis (it is gated on pca_cond being present, exactly as eval
    gates it).  Here we only read the raw config values.
    """
    data_cfg = OmegaConf.load(CONFIG_DATA)
    _nc = data_cfg.get("n_components_cond", None)
    n_comp_cfg = OmegaConf.to_container(_nc, resolve=True) if _nc is not None else None
    _cs = data_cfg.get("cond_smooth_sigma", None)
    cond_smooth_sigma = OmegaConf.to_container(_cs, resolve=True) if _cs is not None else None
    cond_smooth_method = data_cfg.get("cond_smooth_method", "gaussian")
    return n_comp_cfg, cond_smooth_sigma, cond_smooth_method


def resolve_pca_basis(pca_state, name, n_comp_cfg):
    """Replicate eval_aero.main() PCA-basis routing (lines 1811-1921) exactly.

    Returns (exp_pca_cond, N_COMP_COND) — the (pca_objects, n_components_cond)
    pair handed to build_cond_tensor for scenario `name`.

    Branches, mirroring the FIXED eval routing:
      * N_COMP_COND = n_comp_cfg if pca_cond else None     (eval line gate)
      * per_scenario[name]["cond"] if present   (own basis, e.g. ssp370)
        else "fit"                               (OOD → fresh per-scenario fit, e.g. ssp126)
    When the checkpoint has NO PCA at all, pca_cond is None -> N_COMP_COND None
    -> exp_pca_cond stays None -> build_cond_tensor skips PCA.
    """
    pca_cond = pca_state.get("cond") if pca_state else None
    N_COMP_COND = n_comp_cfg if (pca_cond and n_comp_cfg is not None) else None

    pca_per_scenario = pca_state.get("per_scenario") if pca_state else None
    exp_pca_cond = pca_cond
    routing = "reference basis (pca_cond)" if pca_cond is not None else "NO PCA (absent in ckpt)"
    if pca_per_scenario is not None and N_COMP_COND is not None:
        entry = pca_per_scenario.get(name)
        if entry is not None:
            exp_pca_cond = entry.get("cond")
            routing = f"OWN '{name}' scenario basis"
        else:
            # OOD scenario: mirror the FIXED eval routing — fit a fresh
            # per-scenario basis ("fit" sentinel), not the aaer reference.
            exp_pca_cond = "fit"
            routing = f"fit FRESH per-scenario basis (no '{name}' basis, OOD)"
    print(f"    [PCA] {name}: {routing}; N_COMP_COND={N_COMP_COND}")
    return exp_pca_cond, N_COMP_COND


def build_raw_2015_frame(cond_file, time_dim):
    """Pre-normalize control: raw (CO2, SUL) 2015 frame straight from the file.

    Mirrors build_cond_tensor's open/rename/stack but WITHOUT normalize/smooth/
    PCA, so we can confirm whether any diff in the consumed frame is already
    present in the raw inventory or introduced by the pipeline.
    Returns (raw_2015 (C, H, W) float64, years).
    """
    import xarray as xr
    ds = xr.open_dataset(cond_file)
    if time_dim not in ds.dims and "year" in ds.dims:
        ds = ds.rename({"year": time_dim})
    ds = ds[COND_VARS]
    stacked = ds.to_stacked_array("var", sample_dims=[time_dim, "lon", "lat"])
    stacked = stacked.transpose("var", time_dim, "lat", "lon")
    arr = stacked.values.astype(np.float64)            # (C, T, H, W)
    years = extract_years(ds[time_dim].values)
    ds.close()
    return arr, years


def year_index(years, target=2015):
    idx = np.where(years == target)[0]
    if len(idx) == 0:
        raise ValueError(f"year {target} not found in cond years "
                         f"({years[0]}..{years[-1]})")
    return int(idx[0])


def channel_stats(label, a, b, lat=None):
    """Per-channel global-mean / max|diff| / RMS|diff| between two (C,H,W) frames."""
    print(f"  {label}:")
    for c, vname in enumerate(COND_VARS):
        fa, fb = a[c], b[c]
        d = fa - fb
        gm_a = float(np.mean(fa))
        gm_b = float(np.mean(fb))
        maxd = float(np.max(np.abs(d)))
        rmsd = float(np.sqrt(np.mean(d ** 2)))
        print(f"    {vname:4s}  gmean ssp126={gm_a:+.5f}  ssp370={gm_b:+.5f}  "
              f"max|diff|={maxd:.5f}  RMS|diff|={rmsd:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST A — the decisive cond diff (zero GPU)
# ─────────────────────────────────────────────────────────────────────────────
def test_a(pca_state):
    print("\n" + "=" * 72)
    print("TEST A — CONSUMED 2015 cond diff (ssp126 vs ssp370), zero GPU")
    print("=" * 72)

    n_comp_cfg, smooth_sigma, smooth_method = resolve_cond_args()
    print(f"  config_data: n_components_cond={n_comp_cfg}  "
          f"cond_smooth_sigma={smooth_sigma}  method={smooth_method}")

    exp126 = get_experiment("ssp126")
    exp370 = get_experiment("ssp370")

    consumed = {}
    raw_frames = {}
    for exp in (exp126, exp370):
        name = exp["name"]
        pca_obj, n_comp = resolve_pca_basis(pca_state, name, n_comp_cfg)
        cond_tensor, years, lat, lon = build_cond_tensor(
            exp["cond_file"], COND_VARS, exp["time_dim"],
            pca_obj, n_comp, smooth_sigma, smooth_method,
        )
        i2015 = year_index(years, 2015)
        consumed[name] = cond_tensor[:, i2015].numpy()      # (C, H, W) normalised+pca
        raw_arr, raw_years = build_raw_2015_frame(exp["cond_file"], exp["time_dim"])
        ri = year_index(raw_years, 2015)
        raw_frames[name] = raw_arr[:, ri]                   # (C, H, W) raw
        print(f"    {name}: cond_file={os.path.basename(exp['cond_file'])}  "
              f"shape={tuple(cond_tensor.shape)}  2015 idx={i2015}")

    print("\n  --- RAW (pre-normalize) 2015 frame, control ---")
    channel_stats("RAW ssp126 vs ssp370", raw_frames["ssp126"], raw_frames["ssp370"])

    print("\n  --- CONSUMED (normalised [+smooth+PCA]) 2015 frame, what model eats ---")
    channel_stats("CONSUMED ssp126 vs ssp370",
                  consumed["ssp126"], consumed["ssp370"])

    # Verdict on the consumed diff
    cons_max = max(
        float(np.max(np.abs(consumed["ssp126"][c] - consumed["ssp370"][c])))
        for c in range(len(COND_VARS))
    )
    consumed_differ = cons_max > 1e-5
    print("\n  >>> CONSUMED 2015 cond "
          + ("DIFFERS" if consumed_differ else "is IDENTICAL")
          + f" between ssp126 and ssp370 (max|diff| across channels = {cons_max:.2e}).")
    return consumed, consumed_differ


# ─────────────────────────────────────────────────────────────────────────────
# TEST B — same forward path on the two consumed 2015 frames (1 GPU)
# ─────────────────────────────────────────────────────────────────────────────
def test_b(consumed, model, scheduler, device, sample_steps, seed):
    print("\n" + "=" * 72)
    print("TEST B — model output on consumed 2015 frame (ssp126 vs ssp370), 1 GPU")
    print("=" * 72)
    dtype = torch.float32   # match eval's tensor dtype (eval uses autocast bf16;
    # we use fp32 here for a clean deterministic A/B — autocast adds nondeterm.)

    def run_one(frame_chw):
        # build a length-1 cond (C, T=1, H, W) and run the SAME eval sampler.
        cond = torch.from_numpy(frame_chw).float().unsqueeze(1)   # (C, 1, H, W)
        out = generate_timeseries(
            model, scheduler, cond, device, dtype, sample_steps,
            batch_size=1, seed=seed,
        )                                                         # (1, H, W) normalised
        return out[0] * 21.0 + 4.5                                # denorm -> C, (H, W)

    out370 = run_one(consumed["ssp370"])
    out126 = run_one(consumed["ssp126"])

    gm370 = float(np.mean(out370))
    gm126 = float(np.mean(out126))
    maxd = float(np.max(np.abs(out126 - out370)))
    print(f"  ssp370 2015 output  global-mean = {gm370:+.4f} C")
    print(f"  ssp126 2015 output  global-mean = {gm126:+.4f} C")
    print(f"  ssp126 - ssp370     global-mean = {gm126 - gm370:+.4f} C")
    print(f"  max|diff| (ssp126 vs ssp370)    = {maxd:.5f} C")

    # determinism control: ssp370 frame twice, same seed -> diff ~ 0
    out370b = run_one(consumed["ssp370"])
    det = float(np.max(np.abs(out370b - out370)))
    print(f"\n  [determinism control] ssp370 frame run twice, same seed: "
          f"max|diff| = {det:.2e}  "
          + ("(deterministic, OK)" if det < 1e-4 else "(NONDETERMINISTIC!)"))
    return (gm126 - gm370), maxd, det


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="checkpoint .pt on the mount (default: run_mseyb_852.pt)")
    ap.add_argument("--members", type=int, default=1,
                    help="members per scenario (single-member diagnostic; default 1)")
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed RNG seed, identical for both scenarios")
    ap.add_argument("--sample-steps", type=int, default=50,
                    help="diffusion sampling steps (eval default 50)")
    ap.add_argument("--test-a-only", action="store_true",
                    help="run only the GPU-free cond diff (skip model load + Test B)")
    args = ap.parse_args()

    if args.members != 1:
        print(f"[note] --members={args.members} ignored; this is a single-frame "
              f"determinism diagnostic (1 member).")

    print(f"[cfg] checkpoint = {args.checkpoint}")
    print(f"[cfg] seed = {args.seed}  sample_steps = {args.sample_steps}  "
          f"test_a_only = {args.test_a_only}")

    # Load the checkpoint's PCA state for Test A routing. We need the model for
    # Test B; load it once unless --test-a-only (then load PCA state cheaply).
    if args.test_a_only:
        # cheap: read only the PCA blob, no model instantiation / GPU.
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        pca_state = ckpt.get("PCA")
        print(f"[PCA] {'present' if pca_state else 'ABSENT'} in checkpoint "
              f"(keys={list(ckpt.keys())})")
        del ckpt
        consumed, consumed_differ = test_a(pca_state)
        print_decision(consumed_differ, ran_test_b=False)
        return

    if not torch.cuda.is_available():
        sys.exit("[FATAL] Test B needs a GPU but torch.cuda.is_available()=False. "
                 "Re-run with --test-a-only on CPU, or run on a gpu-small node.")
    device = torch.device("cuda")
    model, pca_state = load_model(args.checkpoint, CONFIG_AERO, device)
    model = model.to(torch.float32)
    print(f"[PCA] {'present' if pca_state else 'ABSENT'} in checkpoint")

    cfg = OmegaConf.load(CONFIG_AERO)
    scheduler = instantiate(cfg.scheduler)

    consumed, consumed_differ = test_a(pca_state)
    gm_diff, out_maxd, det = test_b(
        consumed, model, scheduler, device, args.sample_steps, args.seed
    )
    print_decision(consumed_differ, ran_test_b=True,
                   gm_diff=gm_diff, out_maxd=out_maxd, det=det)


def print_decision(consumed_differ, ran_test_b, gm_diff=None, out_maxd=None, det=None):
    print("\n" + "=" * 72)
    print("DECISION")
    print("=" * 72)
    if consumed_differ:
        print("  CONSUMED 2015 cond DIFFERS (Test A) ->")
        print("  CAUSE = cond-pipeline artifact. The model is fed different 2015")
        print("  conditioning for ssp126 vs ssp370 (raw inventory and/or")
        print("  normalize/smooth/PCA routing). Inspect the per-channel RAW vs")
        print("  CONSUMED stats above to localise (raw inventory diff vs")
        print("  PCA-basis routing: ssp126->reference, ssp370->own).")
        return
    # consumed identical
    if not ran_test_b:
        print("  CONSUMED 2015 cond is IDENTICAL (Test A); Test B skipped.")
        print("  Since the model is memoryless per-year, identical cond MUST give")
        print("  identical output -> a current-code cold start is NOT explained by")
        print("  the cond. Run full A+B on a GPU to confirm the forward path is")
        print("  deterministic (if so: cold start is an OLD-SETUP artifact).")
        return
    outputs_differ = (out_maxd is not None and out_maxd > 1e-3)
    if outputs_differ:
        print("  CONSUMED identical (Test A) BUT OUTPUTS DIFFER (Test B) ->")
        print(f"  forward-path nondeterminism/bug (out max|diff|={out_maxd:.4f} C, "
              f"GM diff={gm_diff:+.4f} C, determinism control={det:.2e}).")
    else:
        print("  CONSUMED identical AND OUTPUTS identical ->")
        print(f"  cold start NOT reproducible on current code/cond "
              f"(GM diff={gm_diff:+.4f} C, out max|diff|={out_maxd:.4f} C). It was")
        print("  an OLD-SETUP artifact (stale cond file / pre-fix checkpoint).")


if __name__ == "__main__":
    main()
