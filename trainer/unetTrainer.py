import os
import random
import subprocess
import sys
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
    latitude_weight = torch.cos(latitude_rad).clamp(min=0.2)

    # Weight the loss
    # print(spatial_loss.shape,latitude_weight.shape)
    lat_weighted_loss = torch.einsum('...yx,y->...yx', spatial_loss,
                                     latitude_weight).mean()  # (spatial_loss * latitude_weight).mean()

    return lat_weighted_loss


def calc_mse_loss_precomputed_sq(sq_tensor, lats):
    """Lat-weighted mean of an already-squared tensor.
    Avoids allocating a zeros_like tensor for the null-anomaly baseline loss."""
    latitude = torch.as_tensor(lats.values, dtype=sq_tensor.dtype, device=sq_tensor.device)
    latitude_weight = torch.cos(torch.deg2rad(latitude)).clamp(min=0.2)
    return torch.einsum('...yx,y->...yx', sq_tensor, latitude_weight).mean()


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
        self.val_loader    = None        # set externally in main_aero.py
        self.val_every     = 10          # evaluate held-out members every N epochs
        self.best_val_mse = float("inf")  # tracks best VAL/MSE for auto-save (lower is better)

        self.cond_loss_scaling = 0.0  # always start silent
        self.cond_warmup_epochs = 5   # Phase 1: hold at 0.0 for this many epochs
        self.cond_ramp_epochs   = 30  # Phase 2: linearly ramp 0 → cond_max_scaling over this many epochs
        self.cond_max_scaling   = 0.4 # fixed cap — higher values caused instability
        # CFG dropout prob: fraction of batch where cond_map is zeroed.
        # Eliminates the expensive second out_null forward pass.
        self.cfg_drop_prob = getattr(self, "cfg_drop_prob", 0.1)
        # Per-channel CFG dropout: independently drop CO2 (ch 0) and SUL (ch 1).
        # Required for per-channel classifier-free guidance at inference time.
        # Each channel is dropped independently, so the model sees all four
        # conditioning subsets: (co2, sul), (co2, null), (null, sul), (null, null).
        self.cfg_co2_drop_prob = getattr(self, "cfg_co2_drop_prob", self.cfg_drop_prob)
        self.cfg_sul_drop_prob = getattr(self, "cfg_sul_drop_prob", self.cfg_drop_prob)
        self._cached_sensitivity = 0.0  # last valid sensitivity value

        # ── Adaptive loss scaling ────────────────────────────────────────────
        # When adaptive_loss_scaling=True, cond/tcre loss scalings are updated
        # each sync step so that each component contributes a fixed fraction of
        # the MSE loss in expectation.  EMAs of the raw (unscaled) loss values
        # are used; they are saved/restored with checkpoints.
        self._adaptive_loss_scaling = getattr(self, "adaptive_loss_scaling", False)
        self._cond_target_fraction = getattr(self, "cond_target_fraction", 0.30)
        self._tcre_target_fraction = getattr(self, "tcre_target_fraction", 0.05)
        self._ema_mse  = None   # EMA of mse_loss magnitude
        self._ema_cond = None   # EMA of unscaled cond_loss magnitude
        self._ema_tcre = None   # EMA of unscaled tcre_loss magnitude
        self._ema_decay = 0.99  # ~100 sync-step half-life — slow, stable adjustments

        # ── TCRE regularization ──────────────────────────────────────────────
        # Penalize the model's *per-scenario* forced-response slope for deviating
        # from a linear relation precomputed from CESM2 ensemble-mean training
        # data, separately for each of {hist, ssp370, ghg} (aaer excluded):
        #     gmean(ΔT_norm) ≈ tcre_slopes[sid] * gmean(cumCO2_norm) + tcre_intercepts[sid]
        # The model slope is the batch OLS slope of dT_gmean vs co2_gmean on the
        # rows of that scenario, using pred_anomaly = pred_x0_cond − pred_x0_null
        # (no dedicated CO2-only forward pass).
        self.tcre_loss_scaling = getattr(self, "tcre_loss_scaling", 0.0)
        self.tcre_slopes     = {}  # {scenario_id: float}; filled by _precompute_tcre_slope()
        self.tcre_intercepts = {}
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

        if self.accelerator.is_main_process:
            print(f"[TRAINER] LR: {self.lr:.2e}")
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
        # load_path="0"     → start from scratch (no checkpoint loaded)
        # load_path="newest" → auto-resolve the latest checkpoint in save_dir
        if self.load_path:
            resolved = self._resolve_load_path(self.load_path)
            if resolved:
                self.load_path = resolved
                self.load(resolved)
            else:
                self.load_path = None

        # Prepare everything for GPU training
        self.prepare()

        # Precompute per-scenario TCRE target slopes from CESM2 training data.
        # Checkpoint may override via loaded tcre_slopes/tcre_intercepts — done
        # above in self.load(); we still compute if not restored so new runs
        # get valid targets.
        if self.tcre_loss_scaling > 0 and not self.tcre_slopes:
            self._precompute_tcre_slope()

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

                loss, mse_loss, cond_loss, sens, scen_disc, tcre_loss = self.get_loss(batch, cond, scenario_ids=scenario_ids)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    self.ema_model.update()
                    self._update_loss_emas()
                    self._update_cond_scaling(sens.detach().item(), epoch)

                    if self.accelerator.is_main_process:
                        if self.global_step % self.save_every == 0:
                            self.save(epoch)

                    avg_loss        = self.accelerator.gather_for_metrics(loss).mean()
                    avg_mse_loss    = self.accelerator.gather_for_metrics(mse_loss).mean()
                    avg_cond_loss   = self.accelerator.gather_for_metrics(cond_loss).mean()
                    avg_tcre_loss   = self.accelerator.gather_for_metrics(tcre_loss).mean()
                    avg_sens        = self.accelerator.gather_for_metrics(sens).mean()
                    avg_scen_disc   = self.accelerator.gather_for_metrics(scen_disc).mean()

                    log_dict = {
                        "Training/Loss":  avg_loss.detach().item(),
                        "MSE LOSS":       avg_mse_loss.detach().item(),
                        "COND LOSS":      avg_cond_loss.detach().item(),
                        "TCRE LOSS":      avg_tcre_loss.detach().item(),
                        "SENS":           avg_sens.detach().item(),
                        "SCEN DISC":      avg_scen_disc.detach().item(),
                        "COND SCALE":     self.cond_loss_scaling,
                        "TCRE SCALE":     self.tcre_loss_scaling,
                    }
                    # Per-scenario TCRE target slopes (static once precomputed)
                    if self.tcre_slopes:
                        names = getattr(self.train_set, "scenario_names", None)
                        for sid, s in self.tcre_slopes.items():
                            key = (names[sid] if names and sid < len(names) else f"sid{sid}")
                            log_dict[f"TCRE SLOPE/{key}"] = s

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

        if self.cond_loss_scaling > 0 and self.climatology is not None:
            pred_x0_cond = self.get_original_sample(noisy_samples, model_output, timesteps)

            baseline     = self.climatology.to(device=clean_samples.device, dtype=clean_samples.dtype)
            true_anomaly = self._scenario_ensemble_mean(clean_samples - baseline, scenario_ids)
            pred_anomaly = pred_x0_cond - baseline

            cond_loss   = calc_mse_loss(pred_anomaly, true_anomaly, self._ref_ds.lats)

            scen_disc = torch.zeros(1, device=self.device)
            if scenario_ids is not None:
                unique_scenarios = scenario_ids.unique()
                if len(unique_scenarios) >= 2:
                    scenario_means = [
                        pred_x0_cond[scenario_ids == sid].mean(dim=0)
                        for sid in unique_scenarios
                        if (scenario_ids == sid).sum() > 0
                    ]
                    if len(scenario_means) >= 2:
                        n = len(scenario_means)
                        pair_dists = [
                            (scenario_means[i] - scenario_means[j]).abs().mean()
                            for i in range(n) for j in range(i + 1, n)
                        ]
                        scen_disc = torch.stack(pair_dists).mean()
        else:
            cond_loss   = torch.zeros(1, device=self.device)
            scen_disc   = torch.zeros(1, device=self.device)

        return mse, cond_loss, scen_disc

    def eval_held_out(self, epoch: int) -> None:
        """Evaluate EMA model on held-out members, log VAL/* metrics."""
        if self.val_loader is None:
            return

        self.ema_model.ema_model.eval()

        accum = {"mse": [], "cond": [], "disc": []}

        for batch_tuple in self.val_loader.generate():
            if len(batch_tuple) == 3:
                batch, cond, scenario_ids = batch_tuple
            else:
                batch, cond = batch_tuple
                scenario_ids = None

            mse, cond_l, disc = self._compute_val_metrics(batch, cond, scenario_ids)

            accum["mse"].append(self.accelerator.gather_for_metrics(mse).mean().item())
            accum["cond"].append(self.accelerator.gather_for_metrics(cond_l).mean().item())
            accum["disc"].append(self.accelerator.gather_for_metrics(disc).mean().item())

        import numpy as _np
        val_mse = float(_np.mean(accum["mse"]))

        log_dict = {
            "VAL/MSE":  val_mse,
            "VAL/COND": _np.mean(accum["cond"]),
            "VAL/DISC": _np.mean(accum["disc"]),
        }
        self.accelerator.log(log_dict, step=self.global_step)
        if self.accelerator.is_main_process:
            self.accelerator.print(log_dict, {"Epoch": epoch, "HELD_OUT_VAL": True})

        # ── Auto-save best checkpoint & trigger evaluation ────────────────
        if val_mse < self.best_val_mse and self.accelerator.is_main_process:
            self.best_val_mse = val_mse
            if self.save_name is not None:
                base = self.save_name.split(".pt")[0]
                os.makedirs(self.save_dir, exist_ok=True)
                best_path = os.path.abspath(
                    os.path.join(self.save_dir, f"{base}_best.pt")
                )
                torch.save(
                    {
                        "EMA":                self.ema_model.ema_model.state_dict(),
                        "Unet":               self.accelerator.unwrap_model(self.model).state_dict(),
                        "Optimizer":          self.optimizer.state_dict(),
                        "Global Step":        self.global_step,
                        "cond_loss_scaling":  self.cond_loss_scaling,
                        "tcre_loss_scaling":  self.tcre_loss_scaling,
                        "tcre_slopes":        self.tcre_slopes,
                        "tcre_intercepts":    self.tcre_intercepts,
                        "_ema_mse":           self._ema_mse,
                        "_ema_cond":          self._ema_cond,
                        "_ema_tcre":          self._ema_tcre,
                        "best_val_mse":       val_mse,
                        "best_epoch":         epoch,
                    },
                    best_path,
                    _use_new_zipfile_serialization=False,
                )
                self.accelerator.print(
                    f"  [BEST] New best VAL/MSE={val_mse:.6f} at epoch {epoch} → {best_path}"
                )
                self._spawn_eval(best_path, epoch)

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
            trigger_dir = os.path.join(project_root, "eval_triggers")
            os.makedirs(trigger_dir, exist_ok=True)

            # Write eval output to scratch (writable), not next to the checkpoint
            # which may be in projappl (read-only on compute nodes).
            run_tag_for_dir = os.path.splitext(os.path.basename(self.save_name))[0]
            scratch_root = os.environ.get("SCRATCH", "/scratch/project_462001328")
            output_dir = os.path.join(scratch_root, "eval_output", run_tag_for_dir, f"best_ep{epoch:04d}")

            run_tag = os.path.splitext(os.path.basename(self.save_name))[0]
            trigger_path = os.path.join(trigger_dir, f"eval_request_{run_tag}_ep{epoch:04d}.json")
            payload = {
                "epoch": epoch,
                "checkpoint": checkpoint_path,
                "output_dir": output_dir,
                "sbatch_script": os.path.join(project_root, getattr(self, "eval_script", "run_eval_aero.sh")),
                "log_dir": os.path.join(project_root, "logs"),
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

            # Report the most recent completed eval's TCRE bias table into the
            # training log so the user can track model/CESM2 TCRE drift per
            # checkpoint without digging through separate eval logs. The eval
            # job we just triggered hasn't run yet, so this shows the PREVIOUS
            # milestone's TCRE — one-behind, which is fine.
            self._report_latest_tcre(os.path.dirname(output_dir), current_epoch=epoch)
        except Exception as e:
            self.accelerator.print(
                f"  [EVAL] WARNING: could not write eval trigger for epoch {epoch}: {e}"
            )

    def _report_latest_tcre(self, eval_root: str, current_epoch: int) -> None:
        """Print the newest `tcre_summary.json` found under eval_root to the log."""
        import json, glob
        try:
            candidates = sorted(glob.glob(os.path.join(eval_root, "best_ep*", "tcre_summary.json")))
            if not candidates:
                return
            latest = candidates[-1]
            # extract epoch from the parent dir name (e.g. best_ep0930)
            parent = os.path.basename(os.path.dirname(latest))
            ep_str = parent.replace("best_ep", "").lstrip("0") or "0"
            with open(latest) as f:
                s = json.load(f)
            lines = [f"  [TCRE] latest eval (ep{ep_str}, triggered at ep{current_epoch}):"]
            for sc, v in s.get("per_scenario", {}).items():
                lines.append(
                    f"    {sc:12s} model={v['model_slope']:.4f}  "
                    f"CESM2={v['cesm_slope']:.4f}  ratio={v['ratio']:.3f}  "
                    f"bias={v['bias_pct']:+.1f}%"
                )
            for sc, v in s.get("combined", {}).items():
                lines.append(
                    f"    {sc:12s} model={v['model_slope']:.4f}  "
                    f"CESM2={v['cesm_slope']:.4f}  ratio={v['ratio']:.3f}  "
                    f"bias={v['bias_pct']:+.1f}%"
                )
            self.accelerator.print("\n".join(lines))
        except Exception as e:
            self.accelerator.print(f"  [TCRE] WARNING: failed to load latest tcre_summary: {e}")

    def _update_loss_emas(self) -> None:
        """Update exponential moving averages of unscaled mse, cond, tcre loss."""
        d = self._ema_decay
        mse_val  = getattr(self, "_last_raw_mse",  0.0)
        cond_val = getattr(self, "_last_raw_cond", 0.0)
        tcre_val = getattr(self, "_last_raw_tcre", 0.0)
        self._ema_mse = mse_val if self._ema_mse is None else d * self._ema_mse + (1 - d) * mse_val
        if cond_val > 1e-8:
            self._ema_cond = cond_val if self._ema_cond is None else d * self._ema_cond + (1 - d) * cond_val
        if tcre_val > 1e-8:
            self._ema_tcre = tcre_val if self._ema_tcre is None else d * self._ema_tcre + (1 - d) * tcre_val

    def _update_cond_scaling(self, sensitivity_value: float, epoch: int) -> None:
        """Update cond_loss_scaling on a fixed epoch-based schedule.

        Two phases:
          1. Warmup  (epoch < cond_warmup_epochs):
               scaling = 0.0 — let MSE dominate, build a stable baseline.
          2. Linear ramp then hold  (epoch >= cond_warmup_epochs):
               scaling ramps linearly 0 → cond_max_scaling over cond_ramp_epochs,
               then holds at cond_max_scaling.

        If adaptive_loss_scaling=True, cond and tcre scalings are additionally
        adjusted so each contributes a fixed fraction of the MSE loss.
        """
        # ── Phase 1: warmup ───────────────────────────────────────────────────
        if epoch < self.cond_warmup_epochs:
            self.cond_loss_scaling = 0.0
            return

        # ── Phase 2: linear ramp then hold ────────────────────────────────────
        ramp_epoch = epoch - self.cond_warmup_epochs
        progress = min(ramp_epoch / self.cond_ramp_epochs, 1.0)  # clamp at 1.0 after ramp
        self.cond_loss_scaling = self.cond_max_scaling * progress

        # ── Adaptive scalings ────────────────────────────────────────────────
        if not self._adaptive_loss_scaling:
            return
        if self._ema_mse is None:
            return

        # cond_loss: only adapt once ramped to max; bounded [0.1, cond_max_scaling]
        # to keep scenario separation intact and prevent the instability seen at >0.4
        if self.cond_loss_scaling >= self.cond_max_scaling:
            if self._ema_cond is not None and self._ema_cond > 1e-8:
                target = self._cond_target_fraction * self._ema_mse
                self.cond_loss_scaling = float(max(0.1, min(self.cond_max_scaling, target / self._ema_cond)))

        # tcre_loss: floored at 0.20 (raised from 0.05 after run_slope-tcre ep466
        # plateaued with slopes 50–100% too steep; floor 0.05 was out-competed by cond/MSE).
        if self._ema_tcre is not None and self._ema_tcre > 1e-8:
            target = self._tcre_target_fraction * self._ema_mse
            self.tcre_loss_scaling = float(max(0.20, min(0.5, target / self._ema_tcre)))

    def _precompute_tcre_slope(self) -> None:
        """Fit per-scenario slopes+intercepts of gmean(ΔT_norm) vs gmean(cumCO2_norm).

        One (slope, intercept) pair per scenario in {hist, ssp370, ghg}.  These
        are the CESM2 reference slopes that the model's forced-response slope
        (taken from pred_anomaly = pred_x0_cond - pred_x0_null) must match,
        separately within each scenario's batch subset.  aaer has no CO2
        variation and is excluded.

        Runs on the main process only.  If a scenario can't be loaded its slot
        is left out of the dict; if no scenarios succeed, TCRE is disabled.
        """
        self.tcre_slopes     = {}   # {scenario_id: float}
        self.tcre_intercepts = {}   # {scenario_id: float}

        if not self._multi:
            print("[TRAINER] _precompute_tcre_slope: single-experiment mode — skipping TCRE")
            self.tcre_loss_scaling = 0.0
            return

        scenario_names = self.train_set.scenario_names

        # Area weights (cosine-latitude, clamped) — same as training loss
        lats = torch.as_tensor(self._ref_ds.lats.values, dtype=torch.float32)
        w_lat = torch.cos(torch.deg2rad(lats)).clamp(min=0.2)
        w_lat = w_lat / w_lat.mean()
        w_lat = w_lat.view(1, -1, 1)  # (1, H, 1)

        # Climatology baseline (shared across scenarios)
        if self.climatology is not None:
            clim = self.climatology.detach().to("cpu").squeeze(0).squeeze(1)  # (C, H, W)
        else:
            clim = None

        # hist excluded: tiny ΔT and tiny ΔcumCO2 → noise-dominated regression.
        import numpy as _np
        for name in ("ssp370", "ghg"):
            if name not in scenario_names:
                continue
            sid = scenario_names.index(name)
            ds  = self.train_set.datasets[sid]

            # Load first realization if nothing is loaded yet
            if ds.tensor_data is None or ds.tensor_data_cond is None:
                try:
                    first_real = list(ds.realizations)[0]
                    print(f"[TRAINER] _precompute_tcre_slope: loading {name} realization '{first_real}'")
                    ds.load_data(first_real)
                except Exception as e:
                    print(f"[TRAINER] _precompute_tcre_slope: failed to load {name}: {e}")
                    continue

            t_data = ds.tensor_data
            c_data = ds.tensor_data_cond
            if t_data is None or c_data is None:
                continue

            clim_t = (clim[0] if clim is not None
                      else torch.zeros(t_data.shape[2], t_data.shape[3]))

            dT = t_data[0] - clim_t.unsqueeze(0)                     # (T, H, W)
            dT_gmean  = (dT * w_lat).mean(dim=(1, 2))                # (T,)
            co2_gmean = (c_data[0] * w_lat).mean(dim=(1, 2))         # (T,)

            x = co2_gmean.to(torch.float64).numpy()
            y = dT_gmean.to(torch.float64).numpy()
            if x.size < 3 or float(x.max() - x.min()) < 1e-6:
                print(f"[TRAINER] _precompute_tcre_slope: degenerate CO2 range for {name}")
                continue

            slope, intercept = _np.polyfit(x, y, 1)
            self.tcre_slopes[sid]     = float(slope)
            self.tcre_intercepts[sid] = float(intercept)
            print(
                f"[TRAINER] TCRE precompute [{name} sid={sid}]: slope={slope:.4f} "
                f"intercept={intercept:.4f}  "
                f"(CO2_norm: [{x.min():.3f}, {x.max():.3f}], ΔT_norm: [{y.min():.3f}, {y.max():.3f}])"
            )

        if not self.tcre_slopes:
            print("[TRAINER] _precompute_tcre_slope: no usable scenarios — disabling TCRE")
            self.tcre_loss_scaling = 0.0

    def get_original_sample(self, noisy_sample, model_output, timesteps):
        if isinstance(self.scheduler, ContinuousDDPM):
            return self.scheduler.predict_start_from_v(noisy_sample, timesteps, model_output)
        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        return (alpha_prod_t ** 0.5) * noisy_sample - (beta_prod_t ** 0.5) * model_output

    @staticmethod
    def _scenario_ensemble_mean(anomaly: torch.Tensor, scenario_ids) -> torch.Tensor:
        """Replace each sample with the mean over same-scenario members in the batch.

        This converts a single-realization anomaly (forced signal + internal variability)
        into an estimate of the forced response only, which is learnable from the forcings.
        Falls back to the original tensor when scenario_ids is None or all the same.
        """
        if scenario_ids is None:
            return anomaly
        result = torch.zeros_like(anomaly)
        for sid in scenario_ids.unique():
            mask = scenario_ids == sid
            result[mask] = anomaly[mask].mean(dim=0, keepdim=True)
        return result

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

            NULL_COND_VALUE = -1.0

            # ── Per-channel CFG dropout ───────────────────────────────────────
            # Independently zero CO2 (ch 0) and SUL (ch 1) for randomly chosen
            # batch elements. This trains the model on all four conditioning
            # subsets so that per-channel guidance works correctly at inference.
            if self.cfg_co2_drop_prob > 0 or self.cfg_sul_drop_prob > 0:
                cond_map_input = cond_map.clone()
                if self.cfg_co2_drop_prob > 0:
                    drop_co2 = (
                        torch.rand(clean_samples.shape[0], device=self.device)
                        < self.cfg_co2_drop_prob
                    )
                    cond_map_input[drop_co2, 0] = NULL_COND_VALUE
                if self.cfg_sul_drop_prob > 0:
                    drop_sul = (
                        torch.rand(clean_samples.shape[0], device=self.device)
                        < self.cfg_sul_drop_prob
                    )
                    cond_map_input[drop_sul, 1] = NULL_COND_VALUE
            else:
                cond_map_input = cond_map

            # ── Precompute target and true_anomaly before forward passes ──────
            # This lets us free clean_samples before the (memory-heavy) forward
            # passes, reducing peak VRAM when the null pass is also needed.
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(clean_samples, noise, timesteps)
            else:
                raise NotImplementedError("Only epsilon and v_prediction supported")

            # true_anomaly is only needed on the sync step (where null pass runs).
            # Skip computing it on intermediate accumulation steps to save memory.
            _sync_with_cond = self.cond_loss_scaling > 0 and self.accelerator.sync_gradients
            if _sync_with_cond:
                assert self.climatology is not None, (
                    "climatology must be set on the dataset — "
                    "anomaly loss requires a fixed 1850-1900 baseline"
                )
                baseline = self.climatology.to(device=clean_samples.device, dtype=clean_samples.dtype)
                true_anomaly = self._scenario_ensemble_mean(
                    (clean_samples - baseline).detach(), scenario_ids
                )
                del baseline
            else:
                true_anomaly = None

            del clean_samples  # no longer needed; free before forward passes

            # ── Conditioned forward pass (with gradients) ─────────────────────
            model_output = self.model(noisy_samples, timesteps, cond_map=cond_map_input)

            # ── Primary denoising loss ────────────────────────────────────────
            mse_loss = calc_mse_loss(model_output, target, self._ref_ds.lats)

            # On the final accumulation step (when gradients sync), compute the
            # absolute anomaly loss and TCRE slope-match.  No null forward pass
            # is needed: pred_anomaly is defined as (pred_x0_cond − climatology),
            # not (pred_x0_cond − pred_x0_null).  The CFG-decomposition formulation
            # was structurally biased — pred_x0_null trained toward the data mean
            # rather than the 1850–1900 baseline, which baked a ~+1°C warm offset
            # into every cond_loss gradient.  Anchoring to climatology directly
            # removes that bias source and saves one forward pass per sync step.
            if _sync_with_cond:
                # ── Decode conditioned output to x0 space ─────────────────────
                pred_x0_cond = self.get_original_sample(noisy_samples, model_output, timesteps)

                # ── Predicted anomaly: cond x0 prediction − climatology ───────
                # Absolute anomaly relative to the fixed 1850–1900 baseline,
                # matches what eval_aero plots and what TCRE is computed on.
                clim_b = self.climatology.to(
                    device=pred_x0_cond.device, dtype=pred_x0_cond.dtype
                )
                pred_anomaly = pred_x0_cond - clim_b

                # Sensitivity metric — magnitude of the absolute anomaly.
                cond_sensitivity = pred_anomaly.abs().mean().detach()
                self._cached_sensitivity = cond_sensitivity.item()

                # ── Per-scenario TCRE slope-match on pred_anomaly ─────────────
                # For each scenario in {hist, ssp370, ghg}, fit the batch-OLS
                # slope of gmean(pred_anomaly) vs gmean(cumCO2) across the rows
                # of that scenario, and penalize deviation from the CESM2
                # per-scenario slope precomputed in _precompute_tcre_slope().
                # This is the *total forced response* slope (CO2+SUL), which
                # matches the quantity shown in the eval TCRE figure — no
                # dedicated CO2-only forward pass needed.
                tcre_loss = torch.zeros(1, device=self.device)
                want_tcre = (
                    self.tcre_loss_scaling > 0
                    and bool(self.tcre_slopes)
                    and scenario_ids is not None
                )
                if want_tcre:
                    lats = torch.as_tensor(
                        self._ref_ds.lats.values,
                        dtype=pred_anomaly.dtype,
                        device=pred_anomaly.device,
                    )
                    w = torch.cos(torch.deg2rad(lats)).clamp(min=0.2)
                    w = w / w.mean()                        # (H,)
                    w_b = w.view(1, 1, 1, -1, 1)            # broadcast (B,C,T,H,W)

                    # Per-sample global means (B,)
                    dT_gmean  = (pred_anomaly * w_b).mean(dim=(1, 2, 3, 4))
                    co2_in    = cond_map[:, 0:1]            # (B, 1, T, H, W)
                    co2_gmean = (co2_in * w_b).mean(dim=(1, 2, 3, 4))

                    per_scen_terms = []
                    for sid, slope_ref in self.tcre_slopes.items():
                        sid_mask = scenario_ids == sid
                        if sid_mask.sum() < 3:
                            continue
                        x = co2_gmean[sid_mask]
                        y = dT_gmean[sid_mask]
                        x_c = x - x.mean()
                        y_c = y - y.mean()
                        xx  = (x_c * x_c).sum()
                        if xx > 1e-6:
                            slope_model = (x_c * y_c).sum() / xx
                            slope_loss  = (slope_model - slope_ref) ** 2
                        else:
                            slope_loss = torch.zeros((), device=pred_anomaly.device)

                        # Light anchor on the scenario's target line so a uniform
                        # ΔT offset can't hide a slope bias.
                        intercept_ref = self.tcre_intercepts.get(sid, 0.0)
                        target_dT   = slope_ref * x + intercept_ref
                        anchor_loss = ((y - target_dT) ** 2).mean()

                        per_scen_terms.append(slope_loss + 0.1 * anchor_loss)

                    if per_scen_terms:
                        tcre_loss = torch.stack(per_scen_terms).mean()

                del pred_x0_cond

                # ── Conditioning loss ─────────────────────────────────────────
                # Directly penalise the gap between predicted absolute anomaly
                # (pred_x0_cond − climatology) and the true anomaly
                # (clean_samples − climatology).  Equivalent to x0-space MSE on
                # the absolute prediction, with no broken CFG decomposition.
                cond_loss = calc_mse_loss(pred_anomaly, true_anomaly, self._ref_ds.lats)
                del true_anomaly, pred_anomaly

                # ── Scenario discrimination metric ────────────────────────────
                # Mean pairwise L1 distance between per-scenario mean x0 predictions.
                # Positive only if conditioning produces scenario-specific outputs.
                if scenario_ids is not None:
                    unique_scenarios = scenario_ids.unique()
                    if len(unique_scenarios) >= 2:
                        # Recompute pred_x0_cond for scen_disc (already freed above)
                        # Use model_output which still has grad — detach for metric only
                        pred_x0_scen = self.get_original_sample(
                            noisy_samples, model_output.detach(), timesteps
                        )
                        scenario_means = []
                        for sid in unique_scenarios:
                            mask_s = scenario_ids == sid
                            if mask_s.sum() > 0:
                                scenario_means.append(pred_x0_scen[mask_s].mean(dim=0))
                        del pred_x0_scen
                        if len(scenario_means) >= 2:
                            n = len(scenario_means)
                            pair_dists = [
                                (scenario_means[i] - scenario_means[j]).abs().mean()
                                for i in range(n) for j in range(i + 1, n)
                            ]
                            scen_disc = torch.stack(pair_dists).mean()
                        else:
                            scen_disc = torch.zeros(1, device=self.device)
                    else:
                        scen_disc = torch.zeros(1, device=self.device)
                else:
                    scen_disc = torch.zeros(1, device=self.device)

            elif self.cond_loss_scaling > 0:
                # cond_loss_scaling is active but this is a non-sync accumulation step —
                # skip the null pass entirely.  Use cached sensitivity from last sync step.
                cond_loss        = torch.zeros(1, device=self.device)
                tcre_loss        = torch.zeros(1, device=self.device)
                cond_sensitivity = torch.tensor(self._cached_sensitivity, device=self.device)
                scen_disc        = torch.zeros(1, device=self.device)

            else:
                cond_loss        = torch.zeros(1, device=self.device)
                tcre_loss        = torch.zeros(1, device=self.device)
                cond_sensitivity = torch.zeros(1, device=self.device)
                scen_disc        = torch.zeros(1, device=self.device)

            # Cache raw unscaled loss values for adaptive scaling EMA (before multiply)
            self._last_raw_mse  = mse_loss.detach().item()
            self._last_raw_cond = cond_loss.detach().item()
            self._last_raw_tcre = tcre_loss.detach().item()

            # ── Total loss ────────────────────────────────────────────────────
            loss = (
                mse_loss
                + cond_loss * self.cond_loss_scaling
                + tcre_loss * self.tcre_loss_scaling
            )

            # Scale the loss by cosine-weighted latitude
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return (
            loss, mse_loss,
            cond_loss * self.cond_loss_scaling,
            cond_sensitivity, scen_disc,
            tcre_loss * self.tcre_loss_scaling,
        )

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
                "cond_loss_scaling":  self.cond_loss_scaling,
                "tcre_loss_scaling":  self.tcre_loss_scaling,
                "tcre_slopes":        self.tcre_slopes,
                "tcre_intercepts":    self.tcre_intercepts,
                "_ema_mse":           self._ema_mse,
                "_ema_cond":          self._ema_cond,
                "_ema_tcre":          self._ema_tcre,
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
                if f.startswith(base + "_") and f.endswith(".pt") and not f.endswith("_best.pt")
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
        model.load_state_dict(migrated_sd, strict=True)

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
                ema_model_sd = self._migrate_circular_conv_keys(ema_model_sd, self.accelerator.unwrap_model(self.model))
                ema_model.ema_model.load_state_dict(ema_model_sd)
                ema_model.eval()

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

        # Restore conditioning scale so resume doesn't restart warmup from 0
        self.cond_loss_scaling = checkpoint.get("cond_loss_scaling", 0.0)
        self.tcre_loss_scaling = checkpoint.get("tcre_loss_scaling", self.tcre_loss_scaling)
        self.tcre_slopes       = checkpoint.get("tcre_slopes",     self.tcre_slopes)
        self.tcre_intercepts   = checkpoint.get("tcre_intercepts", self.tcre_intercepts)
        # Drop hist from loaded slopes (now excluded — see _precompute_tcre_slope).
        try:
            scenario_names = self.train_set.scenario_names
            if "hist" in scenario_names:
                hist_sid = scenario_names.index("hist")
                self.tcre_slopes.pop(hist_sid, None)
                self.tcre_intercepts.pop(hist_sid, None)
        except Exception:
            pass
        self._ema_mse  = checkpoint.get("_ema_mse",  None)
        self._ema_cond = checkpoint.get("_ema_cond", None)
        self._ema_tcre = checkpoint.get("_ema_tcre", None)
        print(f"[INFO] Restored cond_loss_scaling={self.cond_loss_scaling:.4f}  "
              f"tcre_loss_scaling={self.tcre_loss_scaling:.4f}  "
              f"tcre_slopes={self.tcre_slopes}")

        # Restore best val MSE so a resumed run doesn't overwrite a better checkpoint
        if "best_val_mse" in checkpoint:
            self.best_val_mse = checkpoint["best_val_mse"]
            print(f"[INFO] Restored best_val_mse={self.best_val_mse:.6f} (epoch {checkpoint.get('best_epoch', '?')})")

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