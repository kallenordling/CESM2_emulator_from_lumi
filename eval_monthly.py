#!/usr/bin/env python3
"""Generate monthly emulator output from a `config_aero_monthly` checkpoint.

eval_aero.py cannot do this. It builds its own conditioning (build_cond_tensor)
on the cond file's ANNUAL axis and feeds the model one frame at a time
(`.unsqueeze(2)` → (B, C, 1, H, W)), and it knows nothing about the
previous-state channel. A monthly checkpoint wants 12-frame windows, cond on a
monthly axis, and a 4th cond channel — so this is a separate script rather than
a flag.

WHAT IT REUSES: EvalClimateDataset, which already loads the monthly target
trees, broadcasts the annual cond onto the monthly axis and appends the
previous-state channel. That keeps eval on exactly the training pipeline
instead of a parallel reimplementation — the thing that caused the train↔eval
PCA mismatch before.

THE PREVIOUS-STATE CHANNEL IS THE HARD PART, and --prev-mode picks your poison:

  truth       The channel holds the TRUE previous month, as during training.
              This is teacher forcing: it answers "given last month, predict
              this month", NOT "run the model forward from 2015". Skill here
              is an upper bound and will flatter the model.

  free        Free-running. Each window's channel is seeded from the last
              month the model itself generated, held constant across the
              window. Honest, but the within-window state is a persistence
              guess, because all 12 frames are denoised jointly — there is no
              causal order to feed one frame's output into the next.

  free-refine free, then re-sample the window with the channel set to the
              model's own generated frames, shifted one step. --refine-passes
              controls how many times. Closest to self-consistent rollout that
              joint denoising allows; costs a full sampling pass each time.

Exposure bias is real here: training only ever showed the model true previous
states. Expect free-running output to drift relative to `truth`, and treat the
gap between the two modes as a measurement, not a nuisance.

Example:

    sbatch --account=project_2019839 --partition=gpumedium --ntasks-per-node=1 \
           --gres=gpu:gh200:1 --cpus-per-task=8 --time=02:00:00 \
           run_roihu.sh eval_monthly.py \
             --checkpoint /scratch/project_2019839/runs/0821_0812/run_monthly_bcprect_7.pt \
             --experiments hist --members 3 --max-months 120 \
             --output-dir /scratch/project_2019839/eval_output/monthly/ep7
"""
import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import OmegaConf

import lumi_paths as L
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from data.climate_dataset import EvalClimateDataset, set_minmax_override

NULL_COND = -1.0     # CFG null value, same convention as eval_aero.py


# ── model ────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, model_config, device):
    """Instantiate the UNet, load EMA weights, honour persisted COND_NORM."""
    cfg = L.resolve_cfg(OmegaConf.load(model_config))
    model = instantiate(cfg.model)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[MODEL] checkpoint keys: {list(ckpt.keys())}")
    print(f"[MODEL] global step: {ckpt.get('Global Step', 'n/a')}")

    state = ckpt.get("EMA") or ckpt["Unet"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[MODEL] {len(missing)} missing keys (first: {missing[:3]})")
    if unexpected:
        print(f"[MODEL] {len(unexpected)} unexpected keys (first: {unexpected[:3]})")

    # Clip ranges the run trained with. Without these the cond normalisation
    # silently differs from training — see eval_aero.py's same block.
    cond_norm = ckpt.get("COND_NORM")
    if cond_norm:
        set_minmax_override(cond_norm)
        print("[COND-NORM] using checkpoint-persisted clip ranges")
    else:
        set_minmax_override(None)
        print("[COND-NORM] WARNING: checkpoint has none — recomputing defaults")

    return model.to(device).eval(), cfg, ckpt.get("PCA")


# ── data ─────────────────────────────────────────────────────────────────────
def build_dataset(exp, data_cfg, pca_state, realization=None):
    """One EvalClimateDataset for a scenario, on the checkpoint's PCA basis."""
    kwargs = dict(
        seq_len=int(data_cfg.seq_len),
        target_vars=OmegaConf.to_container(data_cfg.target_vars, resolve=True),
        cond_vars=OmegaConf.to_container(data_cfg.cond_vars, resolve=True),
        n_components_target=data_cfg.get("n_components_target", None),
        n_components_cond=OmegaConf.to_container(data_cfg.n_components_cond, resolve=True),
        cond_smooth_sigma=OmegaConf.to_container(data_cfg.cond_smooth_sigma, resolve=True),
        cond_smooth_method=data_cfg.get("cond_smooth_method", "gaussian"),
        prev_target_channel=bool(data_cfg.get("prev_target_channel", False)),
    )
    rlz = realization or exp["realizations"][0]
    ds = EvalClimateDataset(
        realizations=[rlz], data_dir=exp["data_dir"], cond_file=exp["cond_file"],
        time_dim=exp["time_dim"], **kwargs,
    )
    # The PCA basis must be the one the checkpoint trained with. __init__ has
    # already loaded (and fitted) once, so restore and reload — a re-fit on
    # eval data is exactly the train/eval mismatch that speckled earlier maps.
    if pca_state:
        per = (pca_state.get("per_scenario") or {}).get(exp["scenario_name"])
        ds.set_pca_state(per or {"cond": pca_state.get("cond"),
                                 "target": pca_state.get("target")})
        ds.load_data(rlz)
        print(f"[PCA] {exp['scenario_name']}: restored from checkpoint")
    return ds, rlz


# ── sampling ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_window(model, scheduler, cond_win, n_members, out_channels,
                  sample_steps, device, amp_dtype, guidance, n_cond_emission):
    """Denoise one (C, F, H, W) window into (members, out_channels, F, H, W).

    Members are the batch dimension: one forward pass covers the ensemble.
    """
    # Tensors stay fp32 and AUTOCAST does the mixed precision. Casting the
    # inputs to bf16 instead trips "Input type (c10::BFloat16) and bias type
    # (float) should be the same" — the weights are fp32.
    C, F, H, W = cond_win.shape
    cond_b = cond_win.unsqueeze(0).expand(n_members, -1, -1, -1, -1).to(device)

    scheduler.set_timesteps(sample_steps)
    steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
    gen = torch.randn(n_members, out_channels, F, H, W, device=device)
    amp = (torch.autocast(device.type, dtype=amp_dtype)
           if amp_dtype is not None else contextlib.nullcontext())

    use_cfg = any(w != 1.0 for w in guidance[:n_cond_emission])
    if use_cfg:
        # Per-channel "only" conditionings over the EMISSION channels. The
        # previous-state channel is never nulled: it is model state, not
        # forcing, and nulling it would ask the model to ignore its own history.
        conds = []
        for keep in range(n_cond_emission):
            c = cond_b.clone()
            for other in range(n_cond_emission):
                if other != keep:
                    c[:, other] = NULL_COND
            conds.append(c)
        cond_null = cond_b.clone()
        cond_null[:, :n_cond_emission] = NULL_COND
        n_pass = len(conds) + 1

    for t_idx in scheduler.timesteps:
      t = scheduler.log_snr(steps[t_idx]).expand(n_members)
      with amp:
        if use_cfg:
            gen_r = gen.repeat(n_pass, 1, 1, 1, 1)
            t_r = t.repeat(n_pass)
            preds = model(gen_r, t_r,
                          cond_map=torch.cat(conds + [cond_null], dim=0)
                          ).split(n_members, dim=0)
            pred = preds[-1]
            for p_only, w in zip(preds[:-1], guidance[:n_cond_emission]):
                pred = pred + w * (p_only - preds[-1])

        else:
            pred = model(gen, t, cond_map=cond_b)
      # The scheduler update runs in fp32 outside autocast; pred may be bf16.
      gen = scheduler.step(pred.to(gen.dtype), timestep=t_idx, sample=gen).prev_sample
    return gen.float().cpu()


def rollout(model, scheduler, cond, n_members, out_channels, seq_len, args,
            device, amp_dtype, n_cond_emission, prev_channel):
    """Walk the record in seq_len windows, returning (members, C, T, H, W)."""
    C, T, H, W = cond.shape
    n_win = T // seq_len
    guidance = [args.guidance_co2, args.guidance_sul, args.guidance_bc]
    out = []
    carry = None                       # last generated TREFHT frame

    for w in range(n_win):
        t0 = w * seq_len
        cond_win = cond[:, t0:t0 + seq_len].clone()

        if prev_channel is not None and args.prev_mode != "truth":
            if carry is not None:
                # Free-running: the window opens on the model's own last month
                # and holds it, since joint denoising gives no causal order to
                # feed frame k-1 into frame k.
                cond_win[prev_channel] = carry
            # w == 0 keeps the dataset's value: persistence of the first true
            # field, which is what training used at a record boundary.

        gen = sample_window(model, scheduler, cond_win, n_members, out_channels,
                            args.sample_steps, device, amp_dtype, guidance,
                            n_cond_emission)

        if prev_channel is not None and args.prev_mode == "free-refine":
            for _ in range(args.refine_passes):
                refined = cond_win.clone()
                # Channel k := the model's own frame k-1, member-averaged (the
                # channel is one map, not an ensemble).
                prev_frames = gen[:, 0].mean(0)              # (F, H, W)
                refined[prev_channel, 1:] = prev_frames[:-1]
                if carry is not None:
                    refined[prev_channel, 0] = carry
                gen = sample_window(model, scheduler, refined, n_members,
                                    out_channels, args.sample_steps, device,
                                    amp_dtype, guidance, n_cond_emission)
                cond_win = refined

        carry = gen[:, 0].mean(0)[-1]                        # (H, W)
        out.append(gen)
        print(f"  [WINDOW {w + 1}/{n_win}] months {t0}-{t0 + seq_len - 1}", flush=True)

    return torch.cat(out, dim=2)                             # (M, C, T, H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="configs/config_aero_monthly.yaml")
    ap.add_argument("--data-config", default="configs/config_data_monthly.yaml")
    ap.add_argument("--experiments", nargs="+", default=["hist"])
    ap.add_argument("--realization", default=None,
                    help="default: the first realization of each experiment")
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--max-months", type=int, default=0,
                    help="truncate the record (0 = all) — use for smoke tests")
    ap.add_argument("--prev-mode", choices=["truth", "free", "free-refine"],
                    default="truth", help="see the module docstring")
    ap.add_argument("--refine-passes", type=int, default=1)
    ap.add_argument("--guidance-co2", type=float, default=1.0)
    ap.add_argument("--guidance-sul", type=float, default=1.0)
    ap.add_argument("--guidance-bc", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # autocast dtype, NOT a tensor dtype — see sample_window.
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    os.makedirs(args.output_dir, exist_ok=True)

    model, cfg, pca_state = load_model(args.checkpoint, args.model_config, device)
    scheduler: ContinuousDDPM = instantiate(cfg.scheduler)
    data_cfg = L.resolve_cfg(OmegaConf.load(args.data_config))

    out_channels = int(cfg.model.get("out_channels", 1))
    cond_channels = int(cfg.model.get("cond_channels", 2))
    seq_len = int(data_cfg.seq_len)
    n_cond_emission = len(OmegaConf.to_container(data_cfg.cond_vars, resolve=True))
    prev_channel = (cond_channels - 1
                    if data_cfg.get("prev_target_channel", False) else None)
    print(f"[SETUP] out_channels={out_channels} cond_channels={cond_channels} "
          f"seq_len={seq_len} prev_channel={prev_channel} mode={args.prev_mode}")

    by_name = {e["scenario_name"]: e for e in
               OmegaConf.to_container(data_cfg.experiment_configs, resolve=True)}
    for name in args.experiments:
        if name not in by_name:
            sys.exit(f"[FATAL] no experiment {name!r} in {args.data_config} "
                     f"(have: {', '.join(by_name)})")

        print(f"\n[EXPERIMENT] {name}")
        ds, rlz = build_dataset(by_name[name], data_cfg, pca_state, args.realization)
        cond = ds.tensor_data_cond
        if cond.shape[0] != cond_channels:
            sys.exit(f"[FATAL] cond has {cond.shape[0]} channels, model wants "
                     f"{cond_channels}. prev_target_channel mismatch between "
                     f"{args.data_config} and the checkpoint?")
        if args.max_months:
            cond = cond[:, :args.max_months]
        print(f"[DATA] realization={rlz} cond={tuple(cond.shape)} "
              f"windows={cond.shape[1] // seq_len}")

        gen = rollout(model, scheduler, cond, args.members, out_channels,
                      seq_len, args, device, amp_dtype, n_cond_emission, prev_channel)

        # Denormalise through the dataset so eval and training agree on the
        # transform (PRECT is log1p'd, not just scaled).
        times = ds.xr_data[ds.time_dim].values[:gen.shape[2]]
        members = [ds.convert_tensor_to_xarray(gen[m]).assign_coords(time=times)
                   for m in range(args.members)]
        out = xr.concat(members, dim="member").assign_coords(
            member=np.arange(args.members))
        out.attrs.update(checkpoint=os.path.basename(args.checkpoint),
                         scenario=name, realization=rlz,
                         prev_mode=args.prev_mode, sample_steps=args.sample_steps)
        path = os.path.join(args.output_dir, f"monthly_{name}.nc")
        out.to_netcdf(path)
        print(f"[WROTE] {path}  {dict(out.sizes)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
