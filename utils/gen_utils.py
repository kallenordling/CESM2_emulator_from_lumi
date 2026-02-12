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
):
    """Generate samples from a trained model with optional classifier-free guidance.

    Args:
        guidance_scale: CFG scale. 1.0 = no guidance (normal sampling).
                        >1.0 amplifies conditioning effect (try 1.5-3.0).
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
    use_cfg = guidance_scale != 1.0

    # Prepare null conditioning for classifier-free guidance
    if use_cfg:
        cond_null = torch.zeros_like(cond_map)
        print(f"[GEN_UTILS] Using classifier-free guidance with scale={guidance_scale}")

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

            # Conditioned prediction
            output_cond = model(
                gen_sample,
                t,
                cond_map=cond_map,
            )

            # Classifier-free guidance
            if use_cfg:
                output_uncond = model(
                    gen_sample,
                    t,
                    cond_map=cond_null,
                )
                output = output_uncond + guidance_scale * (output_cond - output_uncond)
            else:
                output = output_cond

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
):
    """Generate samples from a trained model with optional classifier-free guidance."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    print(cond_map.shape, "cond map shape")
    clean_samples = clean_samples[:, 0, :, :, :].unsqueeze(1)
    gen_sample = torch.randn_like(clean_samples)
    gen_sample = gen_sample.to(device=device, dtype=dtype)
    cond_map = cond_map.to(device=device, dtype=dtype)

    use_cfg = guidance_scale != 1.0
    if use_cfg:
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

        output_cond = model(
            gen_sample,
            t,
            cond_map=cond_map,
        )

        if use_cfg:
            output_uncond = model(
                gen_sample,
                t,
                cond_map=cond_null,
            )
            output = output_uncond + guidance_scale * (output_cond - output_uncond)
        else:
            output = output_cond

        gen_sample = scheduler.step(output, timestep=i, sample=gen_sample).prev_sample

    return gen_sample