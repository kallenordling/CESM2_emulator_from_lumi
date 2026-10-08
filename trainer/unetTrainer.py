import lumi_paths as L
import math
import os
import random
from typing import Any, Callable

import torch
from accelerate import Accelerator
from diffusers import SchedulerMixin
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from ema_pytorch import EMA

from data.climate_dataset import (ClimateDataset, ClimateDataLoader,
                                  set_processed_minmax_override)
from data.multi_experiment_dataset import MultiExperimentDataset, MultiExperimentDataLoader
from models.video_net import UNetModel3D
from custom_diffusers.continuous_ddpm import ContinuousDDPM


# =============================================================================
# Module-level constants
# =============================================================================
# Normalised value representing "no forcing" / pre-industrial baseline in cond.
# Used both as the CFG-dropout replacement value during training and as the
# null-cond input for the second forward pass in `get_loss`.
NULL_COND_VALUE = -1.0

# EMA wrapper hyperparameters (ema-pytorch defaults we override).
_EMA_BETA          = 0.9999
_EMA_UPDATE_AFTER  = 100
_EMA_UPDATE_EVERY  = 10


# =============================================================================
# Module-level helpers
# =============================================================================

def _get_ema_state_dict(ema_obj):
    """Return ema_obj's state_dict, supporting ema-pytorch and custom wrappers."""
    if ema_obj is None:
        return None
    if hasattr(ema_obj, "state_dict"):
        return ema_obj.state_dict()
    if hasattr(ema_obj, "ema_model") and hasattr(ema_obj.ema_model, "state_dict"):
        return ema_obj.ema_model.state_dict()
    raise AttributeError("EMA object does not expose a state_dict().")


def _load_ema_state_dict(ema_obj, state):
    """Load state into an existing EMA object (ema-pytorch or custom wrapper)."""
    if ema_obj is None or state is None:
        return
    if hasattr(ema_obj, "load_state_dict"):
        ema_obj.load_state_dict(state)
        return
    if hasattr(ema_obj, "ema_model") and hasattr(ema_obj.ema_model, "load_state_dict"):
        ema_obj.ema_model.load_state_dict(state)
        return
    raise AttributeError("EMA object does not support load_state_dict().")


def calc_mse_loss(model_output, target, lats, channel_weights=None):
    """Cosine-latitude-weighted MSE.

    Per-pixel squared error is weighted by cos(latitude), clamped at 0.2 so
    poles still contribute. Returns scalar mean over batch / time / lat / lon.

    ``channel_weights``: optional broadcast-ready tensor (e.g. (1, C, 1, 1, 1))
    multiplying the per-channel squared error before the mean — used to down-
    weight the noisier precip channel. None → equal weighting (unchanged).
    """
    spatial_loss = (model_output - target) ** 2
    if channel_weights is not None:
        spatial_loss = spatial_loss * channel_weights.to(
            dtype=spatial_loss.dtype, device=spatial_loss.device,
        )
    latitude = torch.as_tensor(
        lats.values, dtype=spatial_loss.dtype, device=spatial_loss.device,
    )
    latitude_weight = torch.cos(torch.deg2rad(latitude)).clamp(min=0.2)
    return torch.einsum('...yx,y->...yx', spatial_loss, latitude_weight).mean()


def _area_weights_1d(lats, *, dtype, device):
    """1-D area weights ∝ cos(lat), clamped at 0.2 and mean-normalised.

    Most aux losses inside `get_loss` need exactly this 1-D `(H,)` tensor —
    they reshape it to a broadcast-ready form themselves. Keeping the helper
    1-D (vs returning a pre-shaped 5-D tensor) keeps callers explicit about
    the broadcast they need and avoids a hidden shape contract.
    """
    lat = torch.as_tensor(lats.values, dtype=dtype, device=device)
    w = torch.cos(torch.deg2rad(lat)).clamp(min=0.2)
    return w / w.mean()


def _cd_processed_state():
    """Processed cond anchors in force, for COND_PROCESSED_NORM. Import is local
    because climate_dataset sets them lazily at first load_data."""
    try:
        from data.climate_dataset import get_processed_minmax_state
        return get_processed_minmax_state()
    except Exception:
        return None


class UNetTrainer:
    """Trainer class for 2D diffusion models."""

    # State-dict fields persisted to disk on save(), in addition to the
    # always-present "EMA" / "Unet" / "Optimizer" / "Global Step". The aux-loss
    # scalings and their EMAs are gone with the losses themselves; what remains
    # is the conditioning state eval must reproduce exactly.
    _PERSISTED_FIELDS = ()

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
        # Hyperparameters → self.* attributes (lr, save_name, load_path, …).
        self.save_hyperparameters(hyperparameters)

        self.accelerator = accelerator
        self.device      = accelerator.device
        self.weight_dtype = torch.float32
        self.val_set = 0
        self.model = model
        self.scheduler: SchedulerMixin = scheduler

        # ── Order of remaining init steps matters: ───────────────────────────
        # 1. detect multi vs legacy dataloader mode      → self.train_loader/_ref_ds
        # 2. loss schedule + CFG dropout flags           → pure attribute assignments
        # 3. register EBM scalars on self.model          → MUST precede optimizer
        # 4. scheduler.set_timesteps                     → sampler bookkeeping
        # 5. EMA wrapper around self.model               → uses model on device
        # 6. climatology buffer                          → uses self._ref_ds + self.device
        # 7. optimizer                                   → uses self.model.parameters()
        # 8. legacy ClimateDataLoader (multi mode = no-op)
        # 9. step counters + log hparams                 → uses self.train_loader
        # 10. resolve + load checkpoint                  → uses optimizer + model
        # 11. accelerator.prepare                        → wraps model + optimizer
        self._init_dataloader_mode(train_set)
        self._init_loss_schedule_and_flags()

        self.scheduler.set_timesteps(self.sample_steps)
        self.ema_model = EMA(
            self.model,
            beta=_EMA_BETA,
            update_after_step=_EMA_UPDATE_AFTER,
            update_every=_EMA_UPDATE_EVERY,
        ).to(self.device)

        self._init_climatology_buffer()

        if self.accelerator.is_main_process:
            print(f"[TRAINER] LR: {self.lr:.2e}")
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)

        if not self._multi:
            # Legacy single-experiment: build ClimateDataLoader from config callable.
            # Multi-experiment mode supplies the loader pre-built via train_set.
            self.train_loader: ClimateDataLoader = dataloader(
                self.train_set, self.accelerator, self.batch_size, stratified=True,
            )

        self._init_step_counters()
        if self.accelerator.is_main_process:
            self.log_hparams()

        # Resume from checkpoint if one matches load_path (handles "0", "newest", path).
        if self.load_path:
            resolved = self._resolve_load_path(self.load_path)
            if resolved:
                self.load_path = resolved
                self.load(resolved)
            else:
                self.load_path = None

        self.prepare()  # accelerator.prepare(model, optimizer) + optional torch.compile

        # Which scenarios the anchors were fitted on — persisted so eval
        # cannot refit them on a different pool.
        try:
            from data.climate_dataset import get_anchor_scenarios
            self._cond_anchors_state = get_anchor_scenarios()
            ds0 = getattr(self.train_set, "datasets", [self.train_set])[0]
            sig = getattr(ds0, "cond_smooth_sigma", None)
            self._cond_sigma_state = (None if sig is None
                                      else [float(x) for x in sig])
        except Exception as e:
            print(f"[TRAINER] WARNING: could not capture cond-anchor state: {e}")
            self._cond_anchors_state = None

    # ── __init__ helpers ─────────────────────────────────────────────────────

    def _init_dataloader_mode(self, train_set) -> None:
        """Detect multi-experiment vs legacy single-experiment mode.

        Multi-experiment: `train_set` is a MultiExperimentDataLoader built in
        main_aero.py — we use it directly. Legacy: `train_set` is a single
        ClimateDataset, and the dataloader callable from the config will wrap
        it later in __init__.
        """
        if isinstance(train_set, MultiExperimentDataLoader):
            self.train_loader = train_set
            self.train_set    = train_set.dataset
            self._ref_ds      = train_set.dataset.datasets[0]
            self._multi       = True
            print(
                f"[TRAINER] Multi-experiment mode  "
                f"scenarios={self.train_set.scenario_names}"
            )
        else:
            self.train_set = train_set
            self._ref_ds   = train_set
            self._multi    = False

    def _init_loss_schedule_and_flags(self) -> None:
        """Validation bookkeeping, CFG dropout probabilities, and per-channel
        MSE weights. Training is pure denoising MSE: the cond / TCRE / EBM /
        interaction / global-mean / sampled-gain losses and their adaptive
        scaling were removed for the release, having been held at zero by
        mse_only in every run this code is released for.
        """
        # ── Held-out validation bookkeeping ──
        self.val_loader     = None        # set externally in main_aero.py
        self.val_every      = 10          # eval every N epochs
        self.best_val_skill = -float("inf")
        # Periodic force-eval (bypasses the best-skill gate so evals keep firing
        # past the VAL/Skill plateau). 0 = off. Set via config force_eval_every.
        self.force_eval_every       = int(getattr(self, "force_eval_every", 0))
        self._last_eval_epoch       = -1          # last epoch that triggered ANY eval
        self._last_force_eval_epoch = -10**9      # last epoch a force-eval fired

        # ── Per-channel CFG dropout (independent CO2 / SUL / BC) ──
        # Drops cond_map channels to NULL_COND_VALUE for a random fraction of
        # batch elements. Independence is required so per-channel guidance at
        # inference can isolate either pathway.
        self.cfg_drop_prob     = getattr(self, "cfg_drop_prob",     0.1)
        self.cfg_co2_drop_prob = getattr(self, "cfg_co2_drop_prob", self.cfg_drop_prob)
        self.cfg_sul_drop_prob = getattr(self, "cfg_sul_drop_prob", self.cfg_drop_prob)
        # BC (cond ch 2): defaults to 0 so 2-channel configs are unaffected.
        self.cfg_bc_drop_prob  = getattr(self, "cfg_bc_drop_prob", 0.0)

        # ── Per-target-channel MSE weights ([TREFHT, PRECT, …]) ──
        # Down-weight the noisier precip channel so it cannot dominate the
        # shared backbone. None or all-1 → equal weighting. Built into a
        # broadcast-ready (1, C, 1, 1, 1) tensor lazily in get_loss.
        _tvw = getattr(self, "target_var_weights", None)
        if _tvw is not None:
            _tvw = [float(w) for w in _tvw]
        self._target_var_weights = _tvw
        self._target_weight_tensor = None   # lazily built (1, C, 1, 1, 1)

    def _init_climatology_buffer(self) -> None:
        """Move the dataset-side 1850-1900 climatology mean to `self.device`.

        Expected shape on the dataset: (1, C, 1, H, W). Older datasets may
        store (C, H, W) — we pad to 5-D for broadcasting. If the dataset has
        no climatology attribute (e.g. cond-only diagnostic mode), the
        anomaly loss falls back to per-batch mean elsewhere.
        """
        if hasattr(self._ref_ds, "climatology") and self._ref_ds.climatology is not None:
            clim = self._ref_ds.climatology.to(dtype=torch.float32)
            if clim.ndim == 3:
                clim = clim.unsqueeze(0).unsqueeze(2)
            self.climatology = clim.to(self.device)
            print(f"[TRAINER] Loaded climatology baseline, shape={self.climatology.shape}")
        else:
            self.climatology = None
            print("[TRAINER] No climatology found on dataset — anomaly loss will use batch mean as baseline.")

    def _lr_for_step(self, step: int) -> float:
        """Learning rate as a PURE FUNCTION of the optimizer step.

        Resume-invariant by construction: global_step is persisted/restored, so
        the LR at a given step is identical whether reached fresh or via resume —
        no scheduler state to serialize. Decays from the base `self.lr` to
        `self.lr_floor` over `self.lr_decay_horizon_steps`. Mode `off` keeps the
        constant base LR (inert, for clean A/B).
        """
        base = self.lr
        mode = getattr(self, "lr_decay", "off")
        if mode == "off" or mode is None or mode is False:
            return base
        horizon = getattr(self, "lr_decay_horizon_steps", 0)
        floor = min(getattr(self, "lr_floor", base), base)  # never decay UP toward a misconfigured floor>base
        if horizon <= 0:
            return base
        progress = min(step / horizon, 1.0)
        if mode == "cosine":
            return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))
        if mode == "linear":
            return max(floor, base - (base - floor) * progress)
        # Unknown mode → fail safe to constant base LR.
        return base

    def _init_step_counters(self) -> None:
        """Compute step counters that depend on the (now-built) dataloader."""
        self.global_step = 0
        self.first_epoch = 0
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
        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(self.model, self.optimizer)

        # torch.compile: 15-30% throughput gain on fixed-shape UNet inputs.
        # Set env var TORCH_COMPILE=0 to disable if ROCm issues arise.
        import os
        if os.environ.get("TORCH_COMPILE", "1") != "0":
            try:
                self.model = torch.compile(self.model, mode="default")
                print("[TRAINER] torch.compile enabled (mode=reduce-overhead)")
            except Exception as e:
                print(f"[TRAINER] torch.compile skipped: {e}")

    def train(self):
        import time
        for epoch in range(self.first_epoch, self.max_epochs):
            epoch_start = time.time()
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

                loss, mse_loss = self.get_loss(batch, cond, scenario_ids=scenario_ids)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    self.ema_model.update()

                    if self.accelerator.is_main_process:
                        if self.global_step % self.save_every == 0:
                            self.save(epoch)

                    # One gather for both scalars rather than two collectives.
                    _metric_avg = self.accelerator.gather_for_metrics(
                        torch.stack([loss.reshape(()), mse_loss.reshape(())]).unsqueeze(0)
                    ).mean(dim=0)
                    avg_loss, avg_mse_loss = _metric_avg.unbind(0)

                    log_dict = {
                        "Training/Loss":  avg_loss.detach().item(),
                        "MSE LOSS":       avg_mse_loss.detach().item(),
                        "LR":             self.optimizer.param_groups[0]["lr"],
                    }

                    # Per-scenario sample counts — useful for verifying mix is working
                    if scenario_ids is not None and self.accelerator.is_main_process:
                        for i, name in enumerate(self.train_set.scenario_names):
                            log_dict[f"batch/{name}"] = (scenario_ids == i).sum().item()

                    self.accelerator.log(log_dict, step=self.global_step)
                    self.accelerator.log({"Epoch": epoch}, step=self.global_step)
                    self.accelerator.print(log_dict, {"Epoch": epoch})

            if self.accelerator.is_main_process:
                epoch_secs = time.time() - epoch_start
                self.accelerator.print(
                    f"[EPOCH {epoch}] duration: {epoch_secs/60:.1f} min  "
                    f"({epoch_secs:.0f}s)  steps: {step+1}"
                )

            # ── Held-out validation every val_every epochs ───────────────
            if epoch % self.val_every == 0:
                self.eval_held_out(epoch)
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _compute_val_metrics(self, batch, cond_map, scenario_ids=None):
        """Forward pass with EMA model — no backward, returns raw metric tensors.

        Uses a fixed low noise level (t=0.05) so that skill/anomaly metrics are
        stable across epochs and reflect actual denoising quality rather than a
        random mix of noise levels (which was the main source of skill-score noise).
        """
        clean_samples = batch.to(self.weight_dtype)
        noise = torch.randn_like(clean_samples)

        # Fixed low timestep: t=0.05 → mostly clean, consistent across epochs
        if isinstance(self.scheduler, ContinuousDDPM):
            t_fixed = torch.full((clean_samples.shape[0],), 0.05, device=self.device)
            timesteps = self.scheduler.log_snr(t_fixed)
        else:
            t_idx = max(1, self.scheduler.config.num_train_timesteps // 20)
            timesteps = torch.full(
                (clean_samples.shape[0],), t_idx, device=self.device,
            ).long()

        noisy_samples = self.scheduler.add_noise(clean_samples, noise, timesteps)
        ema_model = self.ema_model.ema_model

        model_output = ema_model(noisy_samples, timesteps, cond_map=cond_map)

        if self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(clean_samples, noise, timesteps)
        else:
            target = noise

        mse = calc_mse_loss(model_output, target, self._ref_ds.lats)

        return mse

    def eval_held_out(self, epoch: int) -> None:
        """Evaluate EMA model on held-out members, log VAL/* metrics."""
        if self.val_loader is None:
            return

        self.ema_model.ema_model.eval()

        accum = {"mse": [], "cond": [], "signal": [], "error": [], "disc": []}

        for batch_tuple in self.val_loader.generate():
            if len(batch_tuple) == 3:
                batch, cond, scenario_ids = batch_tuple
            else:
                batch, cond = batch_tuple
                scenario_ids = None

            mse = self._compute_val_metrics(batch, cond, scenario_ids)
            accum["mse"].append(self.accelerator.gather_for_metrics(mse).mean().item())

        import numpy as _np
        # VAL/Skill, VAL/DISC and the ANOM_* metrics are gone with the aux
        # branch that produced them. They had already degenerated: with the aux
        # losses off, ANOM_ERROR and ANOM_SIGNAL were both exactly 0 and
        # VAL/Skill read a constant 1.0, which is worse than not reporting it.
        log_dict = {
            "VAL/MSE":         _np.mean(accum["mse"]),
        }
        self.accelerator.log(log_dict, step=self.global_step)
        if self.accelerator.is_main_process:
            self.accelerator.print(log_dict, {"Epoch": epoch, "HELD_OUT_VAL": True})

        # ── Auto-save best checkpoint & trigger evaluation ────────────────
        # Guard against degenerate epoch-0 skill=1.0 (avg_sig≈0 during warm-up)
        if avg_sig > 1e-4 and val_skill > self.best_val_skill and self.accelerator.is_main_process:
            self.best_val_skill = val_skill
            if self.save_name is not None:
                base = self.save_name.split(".pt")[0]
                os.makedirs(self.save_dir, exist_ok=True)
                best_path = os.path.abspath(
                    os.path.join(self.save_dir, f"{base}_best.pt")
                )
                torch.save(
                    self._build_save_dict(
                        extra={"best_val_skill": val_skill, "best_epoch": epoch}
                    ),
                    best_path,
                    _use_new_zipfile_serialization=False,
                )
                self.accelerator.print(
                    f"  [BEST] New best VAL/Skill={val_skill:.4f} at epoch {epoch} → {best_path}"
                )
                self._spawn_eval(best_path, epoch)
                self._last_eval_epoch = epoch

        # ── Periodic force-eval (bypasses the best-skill gate) ────────────────
        # VAL/Skill is an unreliable gate (single-realization internal
        # variability), so evals stop firing once it plateaus — and you then fly
        # blind on sensitivity/bias past the skill peak. Force an eval every
        # force_eval_every epochs on the current state, reusing the rotating
        # {base}_{epoch}.pt checkpoint. Skipped if a best-eval already fired this
        # epoch (it shares the same trigger / output dir).
        fee = int(getattr(self, "force_eval_every", 0))
        if (fee and epoch > 0 and self.accelerator.is_main_process
                and self.save_name is not None
                and epoch != getattr(self, "_last_eval_epoch", -1)
                and epoch - getattr(self, "_last_force_eval_epoch", -10**9) >= fee):
            base = self.save_name.split(".pt")[0]
            ckpt = os.path.abspath(os.path.join(self.save_dir, f"{base}_{epoch}.pt"))
            if not os.path.exists(ckpt):
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(self._build_save_dict(extra={"force_eval_epoch": epoch}),
                           ckpt, _use_new_zipfile_serialization=False)
            self.accelerator.print(
                f"  [FORCE-EVAL] epoch {epoch} (best-skill gate bypassed) → {ckpt}"
            )
            self._spawn_eval(ckpt, epoch)
            self._last_eval_epoch = epoch
            self._last_force_eval_epoch = epoch

        self.ema_model.ema_model.train()

    def _spawn_eval(self, checkpoint_path: str, epoch: int) -> None:
        """Request an evaluation job by writing a trigger file to disk.

        sbatch is not available inside the Singularity container, so instead
        of calling it directly we write a small JSON trigger file into
        eval_triggers/.  An external watcher script (watch_eval_triggers.sh)
        running outside the container picks these up and calls sbatch.
        """
        import json
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Scratch, not <checkout>/eval_triggers: see lumi_paths.TRIGGER_DIR.
            trigger_dir = L.TRIGGER_DIR
            os.makedirs(trigger_dir, exist_ok=True)

            # Write eval output to scratch (writable), not next to the checkpoint
            # which may be in projappl (read-only on compute nodes).
            # L.EVAL_OUT, not SCRATCH: eval results are collected on
            # LUMI_EVAL_PROJECT's scratch (see lumi_paths.py), which is
            # deliberately independent of the project holding the training data.
            run_tag_for_dir = os.path.splitext(os.path.basename(self.save_name))[0]
            output_dir = os.path.join(L.EVAL_OUT, run_tag_for_dir, f"best_ep{epoch:04d}")

            run_tag = os.path.splitext(os.path.basename(self.save_name))[0]
            trigger_path = os.path.join(trigger_dir, f"eval_request_{run_tag}_ep{epoch:04d}.json")
            payload = {
                "epoch": epoch,
                "checkpoint": checkpoint_path,
                "output_dir": output_dir,
                "sbatch_script": os.path.join(project_root, getattr(self, "eval_script", "run_eval_aero.sh")),
                # Scratch, not project_root/logs: /projappl is at quota and a
                # truncated write there makes a completed eval report failure.
                "log_dir": os.environ.get(
                    "LUMI_LOG_DIR",
                    os.path.join(os.path.dirname(L.RUNS_DIR), "logs")),
                # Only set for runs whose model.{in,out,cond}_channels differ
                # from the production default — see eval_model_config comment
                # in configs/config_aero.yaml. null/absent = watcher uses
                # eval_aero.py's own hardcoded defaults (unchanged behavior).
                "model_config": getattr(self, "eval_model_config", None),
                "data_config": getattr(self, "eval_data_config", None),
            }
            # Write atomically: tmp file then rename so watcher never sees a partial file
            tmp = trigger_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, trigger_path)

            self.accelerator.print(
                f"  [EVAL] Trigger written → {trigger_path}\n"
                f"         (run watch_eval_triggers.sh outside container to auto-submit)"
            )
        except Exception as e:
            self.accelerator.print(
                f"  [EVAL] WARNING: could not write eval trigger for epoch {epoch}: {e}"
            )

    def get_original_sample(self, noisy_sample, model_output, timesteps):
        if isinstance(self.scheduler, ContinuousDDPM):
            return self.scheduler.predict_start_from_v(noisy_sample, timesteps, model_output)
        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        return (alpha_prod_t ** 0.5) * noisy_sample - (beta_prod_t ** 0.5) * model_output

    def get_loss(self, batch, cond_map, scenario_ids=None):
        clean_samples = batch.to(self.weight_dtype)

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

            # ── Per-channel CFG dropout ───────────────────────────────────────
            # Independently zero CO2 (ch 0) and SUL (ch 1) for randomly chosen
            # batch elements. This trains the model on all four conditioning
            # subsets so that per-channel guidance works correctly at inference.
            # Only applied to scenarios that vary both channels (hist, ssp370):
            # aaer has constant CO2 and ghg has constant SUL, so dropping the
            # constant channel teaches the model nothing useful and dropping
            # the varying channel collapses the only signal those samples carry.
            cond_dropped = torch.zeros(clean_samples.shape[0], device=self.device, dtype=torch.bool)
            if (self.cfg_co2_drop_prob > 0 or self.cfg_sul_drop_prob > 0
                    or self.cfg_bc_drop_prob > 0):
                cond_map_input = cond_map.clone()
                if scenario_ids is not None:
                    if not hasattr(self, "_joint_scenario_ids"):
                        names = getattr(self.train_set, "scenario_names", [])
                        self._joint_scenario_ids = torch.tensor(
                            [i for i, n in enumerate(names) if n in ("hist", "ssp370")],
                            device=self.device, dtype=scenario_ids.dtype,
                        )
                    joint_mask = torch.isin(scenario_ids, self._joint_scenario_ids)
                else:
                    joint_mask = torch.ones(
                        clean_samples.shape[0], device=self.device, dtype=torch.bool,
                    )
                if self.cfg_co2_drop_prob > 0:
                    drop_co2 = (
                        torch.rand(clean_samples.shape[0], device=self.device)
                        < self.cfg_co2_drop_prob
                    ) & joint_mask
                    cond_map_input[drop_co2, 0] = NULL_COND_VALUE
                    cond_dropped = cond_dropped | drop_co2
                if self.cfg_sul_drop_prob > 0:
                    drop_sul = (
                        torch.rand(clean_samples.shape[0], device=self.device)
                        < self.cfg_sul_drop_prob
                    ) & joint_mask
                    cond_map_input[drop_sul, 1] = NULL_COND_VALUE
                    cond_dropped = cond_dropped | drop_sul
                if self.cfg_bc_drop_prob > 0:
                    # BC (ch 2): same joint_mask (hist+ssp370) as CO2/SUL — aaer has
                    # constant CO2 and ghg has constant aerosols, so dropping BC
                    # there teaches nothing. BC varies in hist/ssp370 → correct set.
                    if cond_map_input.shape[1] < 3:
                        raise ValueError(
                            f"cfg_bc_drop_prob={self.cfg_bc_drop_prob} expects a BC "
                            f"cond channel (index 2) but the batch has only "
                            f"{cond_map_input.shape[1]} cond channels — the data "
                            f"config (cond_vars) and config_aero.yaml disagree. "
                            f"Launching a BC-era model with a pre-BC 2-channel "
                            f"data_config override (e.g. config_data_ybias.yaml) "
                            f"does this; drop the override or set "
                            f"cfg_bc_drop_prob=0."
                        )
                    drop_bc = (
                        torch.rand(clean_samples.shape[0], device=self.device)
                        < self.cfg_bc_drop_prob
                    ) & joint_mask
                    cond_map_input[drop_bc, 2] = NULL_COND_VALUE
                    cond_dropped = cond_dropped | drop_bc
            else:
                cond_map_input = cond_map

            # ── Target ────────────────────────────────────────────────────────
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(clean_samples, noise, timesteps)
            else:
                raise ValueError(
                    f"Unsupported prediction type {self.scheduler.config.prediction_type}"
                )
            del clean_samples, noise

            model_output = self.model(noisy_samples, timesteps, cond_map=cond_map_input)

            # ── Per-target-channel weights, built once the device is known ────
            if self._target_var_weights is not None and self._target_weight_tensor is None:
                C = model_output.shape[1]
                w = list(self._target_var_weights)
                if len(w) != C:
                    raise ValueError(
                        f"target_var_weights has {len(w)} entries but the model "
                        f"predicts {C} channels"
                    )
                if any(abs(x - 1.0) > 1e-12 for x in w):
                    self._target_weight_tensor = torch.tensor(
                        w, device=self.device, dtype=self.weight_dtype,
                    ).view(1, C, 1, 1, 1)
                else:
                    self._target_var_weights = None   # all 1 → no-op path

            # Pure denoising MSE. This IS the training objective: the cond /
            # TCRE / EBM / interaction / global-mean / sampled-gain terms were
            # removed for the release, having been held at zero throughout by
            # mse_only, so `loss` was already identical to `mse_loss`.
            loss = calc_mse_loss(
                model_output, target, self._ref_ds.lats,
                channel_weights=self._target_weight_tensor,
            )

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                # Apply the scheduled LR once per REAL optimizer update. global_step
                # is incremented in train() AFTER get_loss returns, so here it still
                # holds the count of COMPLETED updates — a fixed 1-step lag that is
                # a pure function of the restored global_step, so resume-invariance
                # is preserved. No-op when lr_decay == "off".
                _lr = self._lr_for_step(self.global_step)
                for g in self.optimizer.param_groups:
                    g["lr"] = _lr
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss, loss.detach()

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

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint I/O: shared state-dict builder + restore helper
    # ─────────────────────────────────────────────────────────────────────────

    def _build_save_dict(self, extra: dict | None = None) -> dict:
        """Assemble the on-disk checkpoint payload.

        Always includes EMA / Unet / Optimizer / Global Step and every entry
        from `_PERSISTED_FIELDS`. `extra` is merged in last for site-specific
        keys (e.g. best_val_skill / best_epoch in the held-out best-save).
        """
        sd = {
            "EMA":         self.ema_model.ema_model.state_dict(),
            "Unet":        self.accelerator.unwrap_model(self.model).state_dict(),
            "Optimizer":   self.optimizer.state_dict(),
            "Global Step": self.global_step,
            "COND_ANCHORS":   getattr(self, "_cond_anchors_state", None),
            # The per-channel smoothing actually used. Eval needs it to verify
            # the data config it was handed matches training; without it the
            # check can only be a uniformity heuristic, which wrongly rejects
            # deliberate PER-CHANNEL sigma (e.g. CO2=0, SUL=BC=4).
            "COND_SIGMA":     getattr(self, "_cond_sigma_state", None),
            # The min/max anchors this run actually normalised with. Without
            # this eval REFITS them and trusts the two to agree -- which is
            # exactly how the first normalize_last evals normalised SUL ~6x
            # off in silence.
            "COND_PROCESSED_NORM": _cd_processed_state(),
        }
        for ckpt_key, attr, _ in self._PERSISTED_FIELDS:
            sd[ckpt_key] = getattr(self, attr)
        if extra:
            sd.update(extra)
        return sd

    def _restore_persisted_fields(self, checkpoint: dict) -> None:
        """Restore the per-run fields listed in _PERSISTED_FIELDS.

        That table is EMPTY in this release: the aux-loss scalings and their
        EMAs went with the losses. The method is kept so a checkpoint written
        before the cleanup — which still carries cond_loss_scaling, tcre_*,
        _ema_* and friends — loads without error, those keys simply being
        ignored. Conditioning state is restored separately (COND_ANCHORS,
        COND_PROCESSED_NORM, COND_SIGMA), which is what eval must reproduce.
        """
        for ckpt_key, attr, zero_disables in self._PERSISTED_FIELDS:
            if ckpt_key not in checkpoint:
                continue
            value = checkpoint[ckpt_key]
            if zero_disables and (value is None or value == 0):
                continue
            setattr(self, attr, value)

    def save(self, epoch: int):
        """Persist a numbered checkpoint and prune all but the last 5 numbered ones."""
        if self.save_name is None:
            return

        state_dict = self._build_save_dict()
        os.makedirs(self.save_dir, exist_ok=True)

        base = self.save_name.split(".pt")[0]
        save_path = os.path.join(self.save_dir, f"{base}_{epoch}.pt")
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)

        # ── Rotate: keep the N newest numbered checkpoints (best.pt is excluded) ─
        # N was 5. An eval is triggered at save time but runs only when SLURM
        # schedules it, and a 5-deep window is ~40 min of training here, so
        # eval_ep0010 (job 21919448) started three hours later and died with
        # FileNotFoundError on a checkpoint rotation had already deleted. Each
        # is ~790 MB, so a deeper window costs GB on a scratch with terabytes
        # free, and buys hours of queue tolerance.
        def _epoch_from_name(fname: str) -> int:
            try:
                return int(fname.split("_")[-1].split(".")[0])
            except ValueError:
                return -1

        existing = [
            os.path.join(self.save_dir, f)
            for f in os.listdir(self.save_dir)
            if f.startswith(base + "_") and f.endswith(".pt") and not f.endswith("_best.pt")
        ]
        keep = int(getattr(self, "keep_checkpoints", 20))
        for stale in sorted(existing, key=_epoch_from_name, reverse=True)[keep:]:
            try:
                os.remove(stale)
            except OSError:
                pass

    @staticmethod
    def _migrate_circular_conv_keys(state_dict, model):
        """Remap pre-LonCircularConv3d checkpoint keys to the new naming scheme.

        LonCircularConv3d wraps nn.Conv3d in a .conv attribute, so old keys like
        'input_conv.weight' become 'input_conv.conv.weight'.  We detect mismatches
        by comparing against the current model's parameter names and remap on the fly.
        """
        model_keys = set(model.state_dict().keys())
        new_sd = {}
        for k, v in state_dict.items():
            if k not in model_keys:
                # Try inserting '.conv' before the final '.weight' / '.bias'
                for suffix in (".weight", ".bias"):
                    if k.endswith(suffix):
                        candidate = k[: -len(suffix)] + ".conv" + suffix
                        if candidate in model_keys:
                            k = candidate
                            break
            new_sd[k] = v
        return new_sd

    def _resolve_load_path(self, load_path):
        """Resolve special load_path values to a concrete file path or None.

        "0"      / 0      → None  (train from scratch)
        "newest"          → newest checkpoint in save_dir matching save_name pattern
        anything else     → returned as-is
        """
        if str(load_path) == "0":
            print("[TRAINER] load_path=0 — starting from scratch")
            return None

        if str(load_path).lower() == "newest":
            import glob
            base = self.save_name.split(".pt")[0]
            pattern = os.path.join(self.save_dir, f"{base}_*.pt")
            paths = [p for p in glob.glob(pattern) if not p.endswith("_best.pt")]
            if not paths:
                print(f"[TRAINER] load_path=newest — no checkpoints found in {self.save_dir}, starting from scratch")
                return None

            def _epoch(p):
                try:
                    return int(os.path.basename(p).split("_")[-1].split(".")[0])
                except ValueError:
                    return -1

            newest = max(paths, key=_epoch)
            print(f"[TRAINER] load_path=newest → {newest}")
            return newest

        return load_path

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Restore model — migrate keys if checkpoint predates LonCircularConv3d
        raw_sd = checkpoint["Unet"]
        model = self.accelerator.unwrap_model(self.model)
        migrated_sd = self._migrate_circular_conv_keys(raw_sd, model)
        # strict=False so old checkpoints (pre-EBM scalars) load cleanly; new
        missing, unexpected = model.load_state_dict(migrated_sd, strict=False)
        if missing or unexpected:
            print(f"[INFO] load_state_dict: missing={list(missing)} unexpected={list(unexpected)}")

        # Restore EMA into the EXISTING wrapper created in __init__.
        # The checkpoint stores self.ema_model.ema_model.state_dict() (just the
        # averaged weights, not the EMA wrapper's bookkeeping), so we load it
        # back into self.ema_model.ema_model. Previously this code built a new
        # local EMA wrapper and loaded into it — the wrapper was then discarded
        # so EMA was silently never restored on resume.
        if "EMA" in checkpoint and checkpoint["EMA"] is not None and hasattr(self, "ema_model"):
            try:
                ema_model_sd = self._migrate_circular_conv_keys(
                    checkpoint["EMA"], self.accelerator.unwrap_model(self.model),
                )
                missing, unexpected = self.ema_model.ema_model.load_state_dict(
                    ema_model_sd, strict=False,
                )
                if missing or unexpected:
                    print(f"[INFO] EMA load_state_dict: missing={list(missing)} "
                          f"unexpected={list(unexpected)}")
                print("[INFO] EMA state restored from checkpoint")
            except Exception as e:
                print(f"[WARN] Could not load EMA: {e}")

        # Restore optimizer (optional; skip if reset_optimizer=True to clear Adam momentum)
        if getattr(self, "reset_optimizer", False):
            print("[INFO] reset_optimizer=True — skipping optimizer state restore (fresh Adam momentum)")
        elif "Optimizer" in checkpoint:
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

        # LR-decay resume guard: the schedule is a pure fn of global_step, so a
        # node-count change (which alters steps/epoch and hence the step↔epoch
        # mapping) is silent unless logged. Print the restored step, horizon,
        # progress and resulting LR so a chained run's position on the curve is
        # auditable. Only when decay is active.
        if getattr(self, "lr_decay", "off") not in ("off", None, False):
            _horizon = getattr(self, "lr_decay_horizon_steps", 0)
            _progress = (self.global_step / _horizon) if _horizon > 0 else 0.0
            print(
                f"[INFO] LR-decay resume: mode={self.lr_decay} "
                f"global_step={self.global_step} horizon={_horizon} "
                f"progress={_progress:.4f} lr={self._lr_for_step(self.global_step):.3e}"
            )

        # _PERSISTED_FIELDS is empty now that the aux-loss scalings are gone;
        # the call is kept so an older checkpoint carrying those keys loads
        # without error (they are simply ignored).
        self._restore_persisted_fields(checkpoint)

        # Restore the cond anchors BEFORE any load_data call: cond fields are
        # normalised lazily in load_data, so injecting the checkpoint's
        # COND_PROCESSED_NORM here keeps a resumed run on the exact anchors it
        # was trained with rather than refitting them.
        pnorm = checkpoint.get("COND_PROCESSED_NORM")
        if pnorm:
            set_processed_minmax_override({k: tuple(v) for k, v in pnorm.items()})
            print("[INFO] Restored cond anchors (COND_PROCESSED_NORM) from checkpoint")

        # Restore best val skill so a resumed run doesn't overwrite a better checkpoint
        if "best_val_skill" in checkpoint:
            self.best_val_skill = checkpoint["best_val_skill"]
            print(f"[INFO] Restored best_val_skill={self.best_val_skill:.4f} (epoch {checkpoint.get('best_epoch', '?')})")

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