import os
import random
from typing import Any, Callable

import torch
from accelerate import Accelerator
from diffusers import SchedulerMixin
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import reduce
# import wandb
from ema_pytorch import EMA

# from utils.viz_utils import create_gif
from data.climate_dataset import ClimateDataset, ClimateDataLoader
from data.multi_experiment_dataset import MultiExperimentDataset, MultiExperimentDataLoader
from models.video_net import UNetModel3D
# from utils.gen_utils import generate_samples
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from torch.serialization import add_safe_globals
import ema_pytorch


def _get_ema_state_dict(ema_obj):
    """
    Works for both ema-pytorch.EMA and custom EMA wrappers.
    Tries common attributes in this order.
    """
    if ema_obj is None:
        return None
    # ema-pytorch exposes .state_dict()
    if hasattr(ema_obj, "state_dict"):
        return ema_obj.state_dict()
    # some wrappers store the averaged model as .ema_model
    if hasattr(ema_obj, "ema_model") and hasattr(ema_obj.ema_model, "state_dict"):
        return ema_obj.ema_model.state_dict()
    raise AttributeError("EMA object does not expose a state_dict().")


def _load_ema_state_dict(ema_obj, state):
    """
    Load EMA state into the existing EMA object.
    Supports ema-pytorch.EMA (has .load_state_dict) and
    fallback to .ema_model.load_state_dict.
    """
    if ema_obj is None or state is None:
        return
    if hasattr(ema_obj, "load_state_dict"):
        ema_obj.load_state_dict(state)
        return
    if hasattr(ema_obj, "ema_model") and hasattr(ema_obj.ema_model, "load_state_dict"):
        ema_obj.ema_model.load_state_dict(state)
        return
    raise AttributeError("EMA object does not support load_state_dict().")


def _list_ckpts_sorted(ckpt_dir, pattern="ckpt_epoch_*.pt"):
    import os, re, glob
    paths = glob.glob(os.path.join(ckpt_dir, pattern))  # <-- module.function

    def _key(p):
        m = re.search(r"epoch[_-](\d+)", os.path.basename(p))
        return (int(m.group(1)) if m else -1, os.path.getmtime(p))

    return sorted(paths, key=_key, reverse=True)


def calc_mse_loss(model_output, target, lats):
    """Manually calculate mse loss"""
    spatial_loss = (model_output - target) ** 2

    # Weight the equator more heavily than the poles
    latitude = torch.as_tensor(lats.values, dtype=spatial_loss.dtype, device=spatial_loss.device)

    latitude_rad = torch.deg2rad(latitude)
    latitude_weight = torch.cos(latitude_rad)

    # Weight the loss
    # print(spatial_loss.shape,latitude_weight.shape)
    lat_weighted_loss = torch.einsum('...yx,y->...yx', spatial_loss,
                                     latitude_weight).mean()  # (spatial_loss * latitude_weight).mean()

    return lat_weighted_loss


class UNetTrainer:
    """Trainer class for 2D diffusion models."""

    def __init__(
            self,
            train_set: "MultiExperimentDataLoader | ClimateDataset",
            model: UNetModel3D,
            scheduler: SchedulerMixin,
            accelerator: Accelerator,
            hyperparameters: DictConfig,
            optimizer: Callable[[Any], Optimizer],
            dataloader: Callable[[Any], DataLoader] = None,  # unused in multi-experiment mode
    ) -> None:
        # Assign the hyperparameters to class attributes
        self.save_hyperparameters(hyperparameters)

        self.accelerator = accelerator
        self.val_set = 0
        self.model = model
        self.scheduler: SchedulerMixin = scheduler

        # ── Detect multi-experiment vs legacy single-experiment mode ──────────
        # Multi-experiment: train_set is a MultiExperimentDataLoader, already
        #   fully built in main_aero.py. The `dataloader` config arg is ignored.
        # Legacy: train_set is a ClimateDataset, dataloader callable is used.
        if isinstance(train_set, MultiExperimentDataLoader):
            self.train_loader = train_set
            self.train_set    = train_set.dataset          # MultiExperimentDataset
            self._ref_ds      = train_set.dataset.datasets[0]  # first ClimateDataset for lats/climatology
            self._multi       = True
            print(
                f"[TRAINER] Multi-experiment mode  "
                f"scenarios={self.train_set.scenario_names}"
            )
        else:
            # Legacy single-experiment path — build loader from config callable
            self.train_set = train_set
            self._ref_ds   = train_set
            self._multi    = False
        # ── Adaptive conditioning loss scaling ───────────────────────────────
        # Phase 1 (warmup): scaling held at 0.0 for `cond_warmup_steps` so the
        #   model first learns a solid MSE baseline without interference.
        # Phase 2 (ramp):   linearly ramps from 0 → cond_max_scaling over the
        #   same number of steps.
        # Phase 3 (adaptive): once fully ramped, the scale is nudged up/down
        #   every `cond_adapt_every` steps based on a running EMA of
        #   cond_sensitivity (how much the conditioning actually changes the
        #   model output).  If the model is ignoring conditioning
        #   (sensitivity < target) the scale grows; if it's already responding
        #   well it shrinks back toward the floor.
        self.cond_loss_scaling     = 0.000          # starts at zero
        self.cond_warmup_steps     = getattr(self, "cond_warmup_steps",     50)
        self.cond_max_scaling      = getattr(self, "cond_max_scaling",       3.0)
        self.cond_min_scaling      = getattr(self, "cond_min_scaling",       0.1)
        self.cond_adapt_every      = getattr(self, "cond_adapt_every",       10)
        self.cond_target_sensitivity = getattr(self, "cond_target_sensitivity", 0.01)
        self._cond_sensitivity_ema = None   # lazily initialised on first update
        # CFG dropout prob: fraction of batch where cond_map is zeroed.
        # Eliminates the expensive second out_null forward pass.
        self.cfg_drop_prob = getattr(self, "cfg_drop_prob", 0.1)
        self._cached_sensitivity = 0.0  # last valid sensitivity value
        self.scheduler.set_timesteps(self.sample_steps)

        # Keep track of our exponential moving average weights
        self.ema_model = EMA(
            self.model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10,
        ).to(self.accelerator.device)

        # Assign the device and weight dtype (32 bit for training)
        self.device = self.accelerator.device
        self.weight_dtype = torch.float32

        # ── Anomaly-based conditioning loss ──────────────────────────────────
        # Pre-compute the 1850-1900 climatological mean from the training set
        # and register it as a non-trainable buffer so it moves with the device.
        # Expected shape: (1, C, 1, H, W)  (one value per variable per pixel,
        # broadcast over batch and time dimensions).
        # The dataset is expected to expose `climatology` as a pre-normalised
        # tensor; if it doesn't, we fall back to zeros (no anomaly shift).
        if hasattr(self._ref_ds, "climatology") and self._ref_ds.climatology is not None:
            clim = self._ref_ds.climatology.to(dtype=torch.float32)
            if clim.ndim == 3:
                clim = clim.unsqueeze(0).unsqueeze(2)
            self.climatology = clim.to(self.device)
            print(f"[TRAINER] Loaded climatology baseline, shape={self.climatology.shape}")
        else:
            self.climatology = None
            print("[TRAINER] No climatology found on dataset – anomaly loss will use batch mean as baseline.")

        self.optimizer = optimizer(
            self.model.parameters(), lr=self.lr
        )

        if self._multi:
            # Loader already built in main_aero.py — nothing to do here
            pass
        else:
            # Legacy single-experiment: build ClimateDataLoader from config callable
            self.train_loader: ClimateDataLoader = dataloader(
                self.train_set,
                self.accelerator,
                self.batch_size,
                stratified=True,
            )
        # self.val_loader: ClimateDataLoader = dataloader(
        #    self.val_set,
        #    self.accelerator,
        #    self.batch_size,
        # )

        # Initialize counters
        self.global_step = 0
        self.first_epoch = 0

        # Keep track of important variables for logging
        self.total_batch_size = (
                self.batch_size
                * self.accelerator.num_processes
                * self.accelerator.gradient_accumulation_steps
        )
        self.num_steps_per_epoch = (
                len(self.train_loader)
                // self.accelerator.gradient_accumulation_steps
                // self.accelerator.num_processes
        )
        self.max_train_steps = self.max_epochs * self.num_steps_per_epoch

        # Log to WANDB (on main process only)
        if self.accelerator.is_main_process:
            self.log_hparams()

        # Load model states from checkpoints if they exist
        if self.load_path:
            self.load(self.load_path)

        # Prepare everything for GPU training
        self.prepare()

    def save_hyperparameters(self, cfg: DictConfig) -> None:
        """Saves the hyperparameters as class attributes."""
        for key, value in cfg.items():
            setattr(self, key, value)

    def log_hparams(self):
        """Logs the hyperparameters to WANDB."""
        # run = self.accelerator.get_tracker("wandb").tracker

        hparam_dict = {
            "Number Training Examples": (
                sum(len(ds) * len(ds.realizations) for ds in self.train_set.datasets)
                if self._multi
                else len(self.train_set) * len(self.train_set.realizations)
            ),
            "Number Epochs": self.max_epochs,
            "Batch Size per Device": self.batch_size,
            "Total Train Batch Size (w. distributed & accumulation)": self.total_batch_size,
            "Gradient Accumulation Steps": self.accelerator.gradient_accumulation_steps,
            "Total Optimization Steps": self.max_train_steps,
        }

        # run.config.update(hparam_dict)

    def prepare(self):
        """Just send all relevant objects through the accelerator to be placed on GPU."""
        # torch.compile must happen BEFORE accelerator.prepare() on ROCm.
        # The DDP wrapper added by prepare() injects internal hooks that confuse
        # dynamo when compile runs after — causing signal-11 segfaults on MI250X.
        # Set env var TORCH_COMPILE=0 to disable.
        if os.environ.get("TORCH_COMPILE", "1") != "0":
            try:
                import torch._dynamo
                # Don't let dynamo try to optimise across the DDP boundary —
                # this is what triggers the segfault when gradient checkpointing
                # is also active.
                torch._dynamo.config.optimize_ddp = False
                self.model = torch.compile(self.model, mode="default")
                print("[TRAINER] torch.compile enabled (mode=default, optimize_ddp=False)")
            except Exception as e:
                print(f"[TRAINER] torch.compile skipped: {e}")

        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(self.model, self.optimizer)

    def train(self):
        epoch_anom_signals = []
        epoch_anom_errors = []
        for epoch in range(self.first_epoch, self.max_epochs):
            for step, batch_tuple in enumerate(self.train_loader.generate()):
                self.model.train()

                # Multi-experiment yields (batch, cond, scenario_ids)
                # Legacy single-experiment yields (batch, cond)
                if len(batch_tuple) == 3:
                    batch, cond, scenario_ids = batch_tuple
                else:
                    batch, cond = batch_tuple
                    scenario_ids = None

                # Skip steps until we reach the resumed step
                if (
                        self.load_path
                        and epoch == self.first_epoch
                        and step < self.resume_step
                ):
                    continue

                loss, mse_loss, cond_loss, anom_signal, anom_error, sens = self.get_loss(batch, cond)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    self.ema_model.update()
                    self._update_cond_scaling(sens.detach().item())

                    if self.accelerator.is_main_process:
                        if self.global_step % self.save_every == 0:
                            self.save(epoch)

                    avg_loss        = self.accelerator.gather_for_metrics(loss).mean()
                    avg_mse_loss    = self.accelerator.gather_for_metrics(mse_loss).mean()
                    avg_cond_loss   = self.accelerator.gather_for_metrics(cond_loss).mean()
                    avg_anom_error  = self.accelerator.gather_for_metrics(anom_error).mean()
                    avg_anom_signal = self.accelerator.gather_for_metrics(anom_signal).mean()
                    avg_sens        = self.accelerator.gather_for_metrics(sens).mean()

                    log_dict = {
                        "Training/Loss": avg_loss.detach().item(),
                        "MSE LOSS":      avg_mse_loss.detach().item(),
                        "COND LOSS":     avg_cond_loss.detach().item(),
                        "ANOM ERROR":    avg_anom_error.detach().item(),
                        "ANOM SIGNAL":   avg_anom_signal.detach().item(),
                        "SENS":          avg_sens.detach().item(),
                        "ANOM SKILL":    (1.0 - avg_anom_error / (avg_anom_signal + 1e-6)).item(),
                        "COND SCALE":    self.cond_loss_scaling,
                    }

                    # Per-scenario sample counts — useful for verifying mix is working
                    if scenario_ids is not None and self.accelerator.is_main_process:
                        for i, name in enumerate(self.train_set.scenario_names):
                            log_dict[f"batch/{name}"] = (scenario_ids == i).sum().item()

                    self.accelerator.log(log_dict, step=self.global_step)
                    self.accelerator.log({"Epoch": epoch}, step=self.global_step)
                    self.accelerator.print(log_dict, {"Epoch": epoch})

    def _update_cond_scaling(self, sensitivity_value: float) -> None:
        """Update cond_loss_scaling adaptively based on training progress.

        Three phases:
          1. Warmup  (step < cond_warmup_steps):
               scaling = 0.0 — let MSE dominate, build a stable baseline.
          2. Linear ramp  (cond_warmup_steps ≤ step < 2 * cond_warmup_steps):
               scaling ramps linearly 0 → cond_max_scaling.
          3. Adaptive  (step ≥ 2 * cond_warmup_steps):
               Every `cond_adapt_every` steps the scale is nudged ±5 % based
               on whether the EMA of cond_sensitivity is above or below the
               target.  If sensitivity is low (model ignoring conditioning)
               the scale grows; if sensitivity is healthy it can relax.
        """
        step = self.global_step

        # ── Phase 1: warmup ───────────────────────────────────────────────────
        if step < self.cond_warmup_steps:
            self.cond_loss_scaling = 0.0
            return

        # ── Phase 2: linear ramp ──────────────────────────────────────────────
        ramp_steps = step - self.cond_warmup_steps
        if ramp_steps < self.cond_warmup_steps:
            progress = ramp_steps / self.cond_warmup_steps          # 0 → 1
            self.cond_loss_scaling = self.cond_max_scaling * progress
            return

        # ── Phase 3: adaptive nudge ───────────────────────────────────────────
        # Update EMA of sensitivity (exponential moving average, α = 0.05)
        alpha = 0.05
        if self._cond_sensitivity_ema is None:
            self._cond_sensitivity_ema = sensitivity_value
        else:
            self._cond_sensitivity_ema = (
                (1.0 - alpha) * self._cond_sensitivity_ema + alpha * sensitivity_value
            )

        # Only nudge every `cond_adapt_every` steps to avoid noise
        if step % self.cond_adapt_every != 0:
            return

        ratio = self._cond_sensitivity_ema / (self.cond_target_sensitivity + 1e-8)
        if ratio < 1.0:
            # Sensitivity below target → model is ignoring conditioning → push harder
            self.cond_loss_scaling = min(
                self.cond_max_scaling,
                self.cond_loss_scaling * 1.05,
            )
        else:
            # Sensitivity healthy → can relax a little
            self.cond_loss_scaling = max(
                self.cond_min_scaling,
                self.cond_loss_scaling * 0.98,
            )

        #if self.accelerator.is_main_process:
        #    print(
        #        f"[COND SCALE] step={step}  "
        #        f"sensitivity_ema={self._cond_sensitivity_ema:.6f}  "
        #        f"target={self.cond_target_sensitivity:.6f}  "
        #        f"cond_loss_scaling={self.cond_loss_scaling:.4f}"
        #    )

    def get_original_sample(self, noisy_sample, model_output, timesteps):

        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (alpha_prod_t ** 0.5) * noisy_sample - (beta_prod_t ** 0.5) * model_output

        return pred_original_sample

    def get_loss(self, batch, cond_map):
        clean_samples = batch.to(self.weight_dtype)
        # cond_map = reduce(clean_samples, "b v t h w -> b v 1 h w", "mean").repeat(
        #    1, 1, clean_samples.shape[-3], 1, 1
        # )

        # Sample noise that we'll add to the clean images
        noise = torch.randn_like(clean_samples)

        # If we are doing continuous diffusion, timesteps need to be from 0 - 1
        if isinstance(self.scheduler, ContinuousDDPM):
            timesteps = torch.rand(clean_samples.shape[0], device=self.device)
            timesteps = self.scheduler.log_snr(timesteps)
        else:
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (clean_samples.shape[0],),
                device=self.device,
            ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_samples = self.scheduler.add_noise(clean_samples, noise, timesteps)

        with self.accelerator.accumulate(self.model):
            model_output = self.model(
                noisy_samples,
                timesteps,
                cond_map=cond_map,
            )

            # Make sure to get the right target for the loss
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(clean_samples, noise, timesteps)
            else:
                raise NotImplementedError("Only epsilon and v_prediction supported")

            # ── Primary denoising loss ────────────────────────────────────────
            mse_loss = calc_mse_loss(model_output, target, self._ref_ds.lats)

            # ── Anomaly-based conditioning loss ───────────────────────────────
            # Goal: the model's x0-prediction should reproduce the *forced*
            # climate anomaly (signal relative to the pre-industrial baseline)
            # rather than just the raw field.  Because the forced signal is only
            # ~1-5 % of total variance, targeting anomalies gives a much stronger
            # gradient than targeting raw values.
            #
            # Steps:
            #  1. Decode model output → predicted x0 (pred_original_sample).
            #  2. Subtract the 1850-1900 climatology to get the anomaly.
            #  3. Do the same for the clean target.
            #  4. Penalise the MSE between the two anomaly fields (lat-weighted).

            if self.cond_loss_scaling > 0:
                # ── CFG dropout: zero cond_map for a random subset of the batch ─────
                # Gives conditioned AND unconditioned outputs in ONE forward pass,
                # replacing the expensive second out_null forward pass entirely.
                drop_mask = (
                    torch.rand(cond_map.shape[0], device=self.device) < self.cfg_drop_prob
                )
                cond_map_cfg = cond_map.clone()
                cond_map_cfg[drop_mask] = 0.0

                # Re-run forward pass with CFG dropout applied so grad flows correctly.
                model_output = self.model(noisy_samples, timesteps, cond_map=cond_map_cfg)
                # Recompute MSE loss against the CFG-aware output
                mse_loss = calc_mse_loss(model_output, target, self._ref_ds.lats)

                # ── cond_sensitivity from in-batch contrast (no second fwd pass) ────
                has_cond   = (~drop_mask).any()
                has_uncond = drop_mask.any()
                if has_cond and has_uncond:
                    cond_sensitivity = (
                        model_output[~drop_mask].mean() - model_output[drop_mask].mean()
                    ).abs()
                    self._cached_sensitivity = cond_sensitivity.detach().item()
                else:
                    cond_sensitivity = torch.tensor(self._cached_sensitivity, device=self.device)

                # ── Decode model output to x0 space ──────────────────────────────────
                if isinstance(self.scheduler, ContinuousDDPM):
                    pred_original_sample = self.scheduler.predict_start_from_v(
                        noisy_samples, timesteps, model_output
                    )
                elif self.scheduler.config.prediction_type == "v_prediction":
                    alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
                    beta_prod_t  = 1 - alpha_prod_t
                    pred_original_sample = (
                        (alpha_prod_t ** 0.5) * noisy_samples
                        - (beta_prod_t  ** 0.5) * model_output
                    )
                else:  # epsilon prediction
                    alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
                    beta_prod_t  = 1 - alpha_prod_t
                    pred_original_sample = (
                        noisy_samples - (beta_prod_t ** 0.5) * model_output
                    ) / (alpha_prod_t ** 0.5)

                # ── Unconditioned x0 from zero-cond samples already in this batch ───
                # No second forward pass needed: the drop_mask samples ARE the
                # unconditioned run.  Use their mean x0 as the model-predicted baseline.
                if has_uncond:
                    pred_baseline = pred_original_sample[drop_mask].mean(dim=0, keepdim=True)
                    pred_baseline = pred_baseline.expand_as(pred_original_sample)
                else:
                    # All samples were conditioned this step; use batch mean as fallback.
                    pred_baseline = pred_original_sample.mean(dim=0, keepdim=True).expand_as(
                        pred_original_sample
                    )

                # ── Climatological baseline for clean anomaly ─────────────────────────
                if self.climatology is not None:
                    baseline = self.climatology.to(
                        device=clean_samples.device, dtype=clean_samples.dtype
                    )
                else:
                    baseline = clean_samples.mean(dim=2, keepdim=True)

                # ── Anomalies ──────────────────────────────────────────────────────────
                clean_anomaly = clean_samples - baseline
                pred_anomaly  = pred_original_sample - pred_baseline

                # ── Contrastive cond_loss ──────────────────────────────────────────────
                mse_null_anomaly    = calc_mse_loss(
                    torch.zeros_like(pred_anomaly), clean_anomaly, self._ref_ds.lats
                )
                mse_correct_anomaly = calc_mse_loss(pred_anomaly, clean_anomaly, self._ref_ds.lats)
                cond_loss = torch.relu(0.01 + mse_correct_anomaly - mse_null_anomaly)

                # ── Anomaly diagnostics ────────────────────────────────────────────────
                anom_signal = clean_anomaly.abs().mean()
                anom_error  = (pred_anomaly - clean_anomaly).abs().mean()
            else:
                cond_loss = torch.zeros(1, device=self.device)
                anom_error =  torch.zeros(1, device=self.device)
                anom_signal =  torch.zeros(1, device=self.device)
                cond_sensitivity =  torch.zeros(1, device=self.device)

            # ── Total loss ────────────────────────────────────────────────────
            loss = mse_loss + cond_loss * self.cond_loss_scaling

            # Scale the loss by cosine-weighted latitude
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss,mse_loss,cond_loss *self.cond_loss_scaling,anom_signal, anom_error,cond_sensitivity

    @torch.inference_mode()
    def validation_loop(self, sanity_check=False) -> None:
        """Runs a single epoch of validation.

        Updates the loss, logs it, and backpropagates the error.
        """
        self.model.eval()
        val_loss = 0

        for batch_idx, batch in enumerate(self.val_loader.generate()):
            # If we are sanity checking, only run 10 batches
            if sanity_check and batch_idx > 10:
                return

            val_loss += self.model_forward_pass(batch)[0].item()

        # Log the average
        self.accelerator.log(
            {"Validation/Loss": val_loss / len(self.val_loader)}, step=self.global_step
        )

    @torch.inference_mode()
    def sample(self) -> None:
        """Samples a batch of images from the model."""

        self.ema_model.eval()
        # Grab a random sample from validation set
        batch = random.choice(self.val_set).unsqueeze(0).to(self.accelerator.device)

        clean_samples = batch.to(self.weight_dtype)

        # Generate the samples
        gen_sample = generate_samples(
            clean_samples, self.scheduler, self.sample_steps, self.ema_model
        )

        # Turn the samples into xr datasets
        gen_ds = self.val_set.convert_tensor_to_xarray(gen_sample[0])
        val_ds = self.val_set.convert_tensor_to_xarray(clean_samples[0])

        # Create a gif of the samples
        gen_frames = create_gif(gen_ds)
        val_frames = create_gif(val_ds)

        # Log the gif to wandb
        for var, gif in gen_frames.items():
            self.accelerator.log(
                {f"Generated {var}": wandb.Video(gif, fps=4)}, step=self.global_step
            )

        for var, gif in val_frames.items():
            self.accelerator.log(
                {f"Original {var}": wandb.Video(gif, fps=4)}, step=self.global_step
            )

    def save(self, epoch: int):
        """Saves the state of training to disk."""
        if self.save_name is None:
            return
        else:
            state_dict = {
                "EMA": self.ema_model.ema_model.state_dict(),
                "Unet": self.accelerator.unwrap_model(self.model).state_dict(),
                "Optimizer": self.optimizer.state_dict(),
                "Global Step": self.global_step,
            }

            # If the directory doesn't exist already create it
            os.makedirs(self.save_dir, exist_ok=True)

            # Create the save filename and add the epoch number
            save_name = self.save_name.split(".pt")[0] + f"_{epoch}.pt"

            # Save the State dictionary to disk
            torch.save(state_dict, os.path.join(self.save_dir, save_name), _use_new_zipfile_serialization=False)

            base = self.save_name.split(".pt")[0]
            save_name = f"{base}_{epoch}.pt"
            save_path = os.path.join(self.save_dir, save_name)

            all_ckpts = [
                os.path.join(self.save_dir, f)
                for f in os.listdir(self.save_dir)
                if f.startswith(base + "_") and f.endswith(".pt")
            ]

            # Sort by epoch number extracted from filename
            def extract_epoch(fname):
                try:
                    return int(fname.split("_")[-1].split(".")[0])
                except ValueError:
                    return -1  # fallback, should not happen

            all_ckpts_sorted = sorted(all_ckpts, key=extract_epoch, reverse=True)

            # Keep last 5, delete the rest
            keep_last = 5
            for ckpt in all_ckpts_sorted[keep_last:]:
                try:
                    os.remove(ckpt)
                except OSError:
                    pass

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Restore model
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["Unet"], strict=True)

        # Restore EMA (optional)
        if "EMA" in checkpoint and checkpoint["EMA"] is not None and hasattr(self, "ema_model"):
            try:
                # self.ema_model.load_state_dict(checkpoint["EMA"], strict=False)
                # self.ema_model = checkpoint["EMA"].to(self.device)

                ema_model_sd = checkpoint["EMA"]  # full EMA state dict (online_model + ema_model)

                # Extract only EMA weights and strip "ema_model." prefix
                # ema_model_sd = {
                #     k.replace("ema_model.", ""): v
                #     for k, v in ema_wrapped_sd.items()
                #     if k.startswith("ema_model.")
                # }

                ema_model = EMA(
                    self.model,
                    beta=0.9999,  # exponential moving average factor
                    update_after_step=100,  # only after this number of .update() calls will it start updating
                    update_every=10,
                ).to(self.device)
                ema_model.ema_model.load_state_dict(ema_model_sd)
                ema_model.eval()

            except Exception as e:
                print(f"[WARN] Could not load EMA: {e}")

        # Restore optimizer (optional)
        if "Optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["Optimizer"])
            except Exception as e:
                print(f"[WARN] Could not load optimizer state: {e}")

        # Restore global step
        self.global_step = checkpoint.get("Global Step", 0)
        print(self.global_step, self.accelerator.gradient_accumulation_steps)
        self.resume_global_step = (
                self.global_step * self.accelerator.gradient_accumulation_steps
        )

        # Avoid ZeroDivisionError if dataloader not yet initialized
        steps_per_epoch_accum = self.num_steps_per_epoch * self.accelerator.gradient_accumulation_steps
        if steps_per_epoch_accum > 0:
            self.resume_step = self.resume_global_step % steps_per_epoch_accum
        else:
            self.resume_step = 0

        # Read first_epoch from the checkpoint filename (pattern: {base}_{epoch}.pt)
        try:
            epoch_from_filename = int(os.path.basename(path).split("_")[-1].split(".")[0])
            self.first_epoch = epoch_from_filename + 1  # resume from the NEXT epoch
            print(f"[INFO] Resuming from epoch {self.first_epoch} (parsed from filename)")
        except (ValueError, IndexError):
            # Fallback: derive from global_step if filename parsing fails
            if self.num_steps_per_epoch > 0:
                self.first_epoch = self.global_step // self.num_steps_per_epoch
            else:
                self.first_epoch = 0
            print(f"[WARN] Could not parse epoch from filename, defaulting to epoch {self.first_epoch}")

        print(f"[INFO] Loaded checkpoint from {path} (step {self.global_step})")