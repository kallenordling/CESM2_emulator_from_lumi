import torch
from torch import Tensor
from tqdm import tqdm
from diffusers import DDPMScheduler
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from einops import reduce


def generate_samples2(
    clean_samples: Tensor,
    cond_map: Tensor,
    scheduler: DDPMScheduler,
    sample_steps: int,
    model: torch.nn.Module,
    disable=False,
    explain=True,
    guidance_scale=1.0,
    guidance_co2: float | None = None,
    guidance_sul: float | None = None,
):
    """Generate samples from a trained model with optional classifier-free guidance.

    Args:
        guidance_scale: Joint CFG scale applied to both channels together.
                        1.0 = no guidance (normal sampling), >1.0 amplifies
                        conditioning effect (try 1.5-3.0).
                        Ignored when guidance_co2 or guidance_sul is set.
        guidance_co2:   Per-channel CFG scale for CO2 (channel 0).
                        When set (along with guidance_sul), enables independent
                        per-channel guidance using 3 model forward passes per step:
                          pred = pred_null
                               + guidance_co2 * (pred_co2_only - pred_null)
                               + guidance_sul * (pred_sul_only - pred_null)
                        Requires the model to have been trained with per-channel
                        CFG dropout (cfg_co2_drop_prob / cfg_sul_drop_prob).
        guidance_sul:   Per-channel CFG scale for SUL (channel 1).
                        See guidance_co2 for details.
    """
    # ===== EMA SAMPLING ASSERT =====
    assert hasattr(model, "parameters"), "model is not a torch module?"

    if hasattr(model, "ema_model"):
        raise RuntimeError(
            "X You passed EMA wrapper instead of ema_model. "
            "Call generate_samples*(..., model=ema.ema_model)"
        )

    print("[GEN_UTILS] Sampling from model:", type(model))
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # clean_samples: (B, C, T, H, W) -> (B, 1, T, H, W)
    clean_samples = clean_samples[:, 0:1, ...]

    # Cond map to device once
    cond_map = cond_map.to(device=device, dtype=dtype, non_blocking=True)

    # Start from pure noise
    gen_sample = torch.randn_like(clean_samples, device=device, dtype=dtype)

    # Timesteps assumed to be set OUTSIDE this function
    timesteps = scheduler.timesteps.to(device)

    # Precompute continuous timesteps if needed
    if isinstance(scheduler, ContinuousDDPM):
        steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        t_all = scheduler.log_snr(steps[timesteps])

    batch_size = clean_samples.shape[0]

    # ── Guidance mode selection ───────────────────────────────────────────────
    # Per-channel CFG takes precedence over joint guidance_scale.
    NULL_COND_VALUE = -1.0
    use_channel_cfg = guidance_co2 is not None or guidance_sul is not None
    use_joint_cfg = (guidance_scale != 1.0) and not use_channel_cfg

    if use_channel_cfg:
        w_co2 = guidance_co2 if guidance_co2 is not None else 1.0
        w_sul = guidance_sul if guidance_sul is not None else 1.0
        # Three null maps for per-channel guidance (all precomputed, no clone per step)
        cond_co2_only = cond_map.clone()
        cond_co2_only[:, 1] = NULL_COND_VALUE   # SUL nulled; CO2 active
        cond_sul_only = cond_map.clone()
        cond_sul_only[:, 0] = NULL_COND_VALUE   # CO2 nulled; SUL active
        cond_null_all = torch.full_like(cond_map, NULL_COND_VALUE)  # both null
        print(
            f"[GEN_UTILS] Per-channel CFG: guidance_co2={w_co2}, guidance_sul={w_sul}"
        )
    elif use_joint_cfg:
        cond_null = torch.zeros_like(cond_map)
        print(f"[GEN_UTILS] Joint CFG: guidance_scale={guidance_scale}")

    # --- Saliency maps (optional) ---
    sal_co2 = None
    sal_sul = None
    if explain:
        cond_map_grad = cond_map.detach().clone().requires_grad_(True)
        gen_sample_fixed = gen_sample.detach().clone()
        t_fixed = scheduler.timesteps[0]

        if isinstance(scheduler, ContinuousDDPM):
            steps_temp = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
            t_explain = scheduler.log_snr(steps_temp[t_fixed]).expand(batch_size)
        else:
            t_explain = t_fixed

        # Forward pass with gradients through the conditioning encoder
        output = model(
            gen_sample_fixed,
            t_explain,
            cond_map=cond_map_grad,
        )

        output_scalar = output.pow(2).mean()
        output_scalar.backward()

        saliency = cond_map_grad.grad.detach()  # [B, 2, T, H, W]
        sal_co2 = saliency[0, 0].abs().mean(dim=0).cpu().numpy()  # [H, W]
        sal_sul = saliency[0, 1].abs().mean(dim=0).cpu().numpy()

    # --- Sampling loop ---
    with torch.inference_mode():
        for step_idx, t_idx in tqdm(
            enumerate(timesteps),
            total=len(timesteps),
            desc="Sampling",
            disable=disable,
        ):
            if isinstance(scheduler, ContinuousDDPM):
                t = t_all[step_idx].expand(batch_size)
            else:
                t = t_idx

            if use_channel_cfg:
                # Per-channel CFG: 3 forward passes
                # pred = pred_null
                #      + w_co2 * (pred_co2_only - pred_null)   ← CO2 contribution
                #      + w_sul * (pred_sul_only - pred_null)   ← SUL contribution
                pred_co2 = model(gen_sample, t, cond_map=cond_co2_only)
                pred_sul = model(gen_sample, t, cond_map=cond_sul_only)
                pred_null = model(gen_sample, t, cond_map=cond_null_all)
                output = pred_null + w_co2 * (pred_co2 - pred_null) + w_sul * (pred_sul - pred_null)
            elif use_joint_cfg:
                # Joint CFG: 2 forward passes
                output_cond = model(gen_sample, t, cond_map=cond_map)
                output_uncond = model(gen_sample, t, cond_map=cond_null)
                output = output_uncond + guidance_scale * (output_cond - output_uncond)
            else:
                # No guidance: 1 forward pass
                output = model(gen_sample, t, cond_map=cond_map)

            # One reverse diffusion step
            gen_sample = scheduler.step(
                output,
                timestep=t_idx,
                sample=gen_sample,
            ).prev_sample

    return gen_sample, sal_co2, sal_sul


@torch.inference_mode()
def generate_samples(
    clean_samples: Tensor,
    cond_map: Tensor,
    scheduler: DDPMScheduler,
    sample_steps: int,
    model: torch.nn.Module,
    disable=False,
    guidance_scale=1.0,
    guidance_co2: float | None = None,
    guidance_sul: float | None = None,
):
    """Generate samples from a trained model with optional classifier-free guidance.

    See generate_samples2 for guidance_co2 / guidance_sul semantics.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    print(cond_map.shape, "cond map shape")
    clean_samples = clean_samples[:, 0, :, :, :].unsqueeze(1)
    gen_sample = torch.randn_like(clean_samples)
    gen_sample = gen_sample.to(device=device, dtype=dtype)
    cond_map = cond_map.to(device=device, dtype=dtype)

    NULL_COND_VALUE = -1.0
    use_channel_cfg = guidance_co2 is not None or guidance_sul is not None
    use_joint_cfg = (guidance_scale != 1.0) and not use_channel_cfg

    if use_channel_cfg:
        w_co2 = guidance_co2 if guidance_co2 is not None else 1.0
        w_sul = guidance_sul if guidance_sul is not None else 1.0
        cond_co2_only = cond_map.clone()
        cond_co2_only[:, 1] = NULL_COND_VALUE
        cond_sul_only = cond_map.clone()
        cond_sul_only[:, 0] = NULL_COND_VALUE
        cond_null_all = torch.full_like(cond_map, NULL_COND_VALUE)
    elif use_joint_cfg:
        cond_null = torch.zeros_like(cond_map)

    scheduler.set_timesteps(sample_steps)

    for i in tqdm(
        scheduler.timesteps,
        "Sampling",
        disable=disable,
    ):
        if isinstance(scheduler, ContinuousDDPM):
            steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=gen_sample.device)
            t = scheduler.log_snr(steps[i]).repeat(clean_samples.shape[0])
        else:
            t = i

        if use_channel_cfg:
            pred_co2 = model(gen_sample, t, cond_map=cond_co2_only)
            pred_sul = model(gen_sample, t, cond_map=cond_sul_only)
            pred_null = model(gen_sample, t, cond_map=cond_null_all)
            output = pred_null + w_co2 * (pred_co2 - pred_null) + w_sul * (pred_sul - pred_null)
        elif use_joint_cfg:
            output_cond = model(gen_sample, t, cond_map=cond_map)
            output_uncond = model(gen_sample, t, cond_map=cond_null)
            output = output_uncond + guidance_scale * (output_cond - output_uncond)
        else:
            output = model(gen_sample, t, cond_map=cond_map)

        gen_sample = scheduler.step(output, timestep=i, sample=gen_sample).prev_sample

    return gen_sample