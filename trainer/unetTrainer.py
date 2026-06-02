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

from data.climate_dataset import ClimateDataset, ClimateDataLoader
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


def calc_mse_loss(model_output, target, lats):
    """Cosine-latitude-weighted MSE.

    Per-pixel squared error is weighted by cos(latitude), clamped at 0.2 so
    poles still contribute. Returns scalar mean over batch / time / lat / lon.
    """
    spatial_loss = (model_output - target) ** 2
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


class UNetTrainer:
    """Trainer class for 2D diffusion models."""

    # State-dict fields persisted to disk on save() and restored on load(),
    # in addition to the always-present "EMA" / "Unet" / "Optimizer" / "Global Step".
    # Each entry: (ckpt_key, attr_name, zero_disables).
    # When zero_disables=True, a restored value of 0 keeps the current
    # (config-provided) attribute instead — protects against a previous run
    # silently disabling the aux loss (e.g. precompute failure) and that 0
    # then sticking forever on subsequent resumes.
    _PERSISTED_FIELDS = (
        # ckpt_key,                    attr_name,                   zero_disables
        ("cond_loss_scaling",          "cond_loss_scaling",         False),
        ("tcre_loss_scaling",          "tcre_loss_scaling",         True),
        ("tcre_slope",                 "tcre_slope",                False),
        ("tcre_intercept",             "tcre_intercept",            False),
        ("_ema_mse",                   "_ema_mse",                  False),
        ("_ema_cond",                  "_ema_cond",                 False),
        ("_ema_tcre",                  "_ema_tcre",                 False),
        ("_ema_ebm",                   "_ema_ebm",                  False),
        ("ebm_loss_scaling",           "ebm_loss_scaling",          False),
        ("_ema_interaction",           "_ema_interaction",          False),
        ("interaction_loss_scaling",   "interaction_loss_scaling",  True),
        ("_ema_gmean",                 "_ema_gmean",                False),
        ("gmean_loss_scaling",         "gmean_loss_scaling",        True),
    )

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
        # 2. loss-schedule + aux-loss flags              → adaptive EMAs, gmean, tcre, ...
        # 3. register EBM scalars on self.model          → MUST precede optimizer
        # 4. scheduler.set_timesteps                     → sampler bookkeeping
        # 5. EMA wrapper around self.model               → uses model on device
        # 6. climatology buffer                          → uses self._ref_ds + self.device
        # 7. optimizer                                   → uses self.model.parameters()
        # 8. legacy ClimateDataLoader (multi mode = no-op)
        # 9. step counters + log hparams                 → uses self.train_loader
        # 10. resolve + load checkpoint                  → uses optimizer + model
        # 11. accelerator.prepare                        → wraps model + optimizer
        # 12. precompute TCRE slope                      → needs train_set, ref_ds
        self._init_dataloader_mode(train_set)
        self._init_loss_schedule_and_flags()
        self._register_ebm_scalars()

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

        # TCRE slopes are deterministic from the train set — recompute if either
        # (a) we just initialised fresh, or (b) we resumed but tcre_slopes wasn't
        # in the checkpoint (the per-scenario dict isn't serialised, only the
        # combined slope/intercept are).
        scen_names = getattr(self.train_set, "scenario_names", [])
        if ((self.tcre_slope is None or not self.tcre_slopes)
                and "hist" in scen_names and "ssp370" in scen_names):
            self._precompute_tcre_slope()

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
        """Initialise the cond-loss warmup/ramp schedule, CFG dropout probs,
        adaptive-loss EMAs, and per-aux-loss configuration flags.

        These are all pure attribute assignments (no side effects on the model)
        and run before EBM parameter registration and the optimizer.
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

        # ── Cond-loss warmup → linear ramp → adaptive (capped) schedule ──
        # Phase 1 (warmup, cond_warmup_epochs):  scaling = 0
        # Phase 2 (ramp, cond_ramp_epochs):      0 → cond_max_scaling
        # Phase 3 (adaptive, after ramp):        nudged toward target_fraction*MSE
        self.cond_loss_scaling   = 0.0
        self.cond_warmup_epochs  = 5
        self.cond_ramp_epochs    = 30
        self.cond_max_scaling    = 0.4    # 1.0 crashed ANOM_SKILL by epoch 6; 0.3 also too high
        self._cached_sensitivity = 0.0
        # MSE-only diagnostic arm: when True, hold cond_loss_scaling=0 so all aux
        # losses (cond/TCRE/low-t/EBM/interaction) + null pass are skipped.
        self.mse_only = bool(getattr(self, "mse_only", False))
        if self.mse_only and getattr(self.accelerator, "is_main_process", True):
            self.accelerator.print("[TRAINER] mse_only=True — aux losses disabled (pure denoising MSE)")

        # ── Per-channel CFG dropout (independent CO2 / SUL) ──
        # Drops cond_map channels to NULL_COND_VALUE for a random fraction of
        # batch elements. Independence is required so per-channel guidance
        # at inference can isolate either pathway.
        self.cfg_drop_prob     = getattr(self, "cfg_drop_prob",     0.1)
        self.cfg_co2_drop_prob = getattr(self, "cfg_co2_drop_prob", self.cfg_drop_prob)
        self.cfg_sul_drop_prob = getattr(self, "cfg_sul_drop_prob", self.cfg_drop_prob)

        # ── Adaptive loss-scaling EMAs (saved/restored with checkpoint) ──
        self._adaptive_loss_scaling = getattr(self, "adaptive_loss_scaling", False)
        self._cond_target_fraction  = getattr(self, "cond_target_fraction",  0.30)
        self._tcre_target_fraction  = getattr(self, "tcre_target_fraction",  0.05)
        self._ema_decay  = 0.99   # ~100-step half-life
        self._ema_mse    = None
        self._ema_cond   = None
        self._ema_tcre   = None
        self._ema_ebm    = None
        self._ema_interaction = None
        self._ema_gmean  = None

        # ── TCRE per-scenario slope-match (precomputed from hist + ssp370) ──
        # tcre_full_anomaly=True scores against (pred − climatology), the
        # quantity eval reports; False scores the CFG decomposition (cond − null).
        self.tcre_loss_scaling = getattr(self, "tcre_loss_scaling", 0.0)
        self.tcre_slope        = None
        self.tcre_intercept    = None
        self.tcre_slopes       = {}          # {scenario_id: (slope, intercept)}
        self.tcre_full_anomaly = getattr(self, "tcre_full_anomaly", False)
        # tcre_lowt: evaluate the TCRE sensitivity constraint on an EXTRA forward
        # pass at a fixed LOW noise level (tcre_lowt_level) instead of the random-t
        # one-step x0. At high t the one-step x0 is a poor estimate, so the random-t
        # TCRE is only a weak proxy the model can satisfy while the *sampled*
        # sensitivity still overshoots (~+13%). At low t one-step x0 ≈ the true
        # sample, so this pins the sampled sensitivity. Adds one conditioned forward
        # (with grad) per optimizer step on the sync iteration; uses full-anomaly
        # (pred − climatology), matching what eval measures.
        self.tcre_lowt       = getattr(self, "tcre_lowt", False)
        self.tcre_lowt_level = float(getattr(self, "tcre_lowt_level", 0.05))

        # ── Superposition (interaction) penalty: f(CO2,SUL) ≈ f(CO2,0)+f(0,SUL)−f(0,0)
        # Applied to hist (low-forcing) samples; ssp370 excluded because the
        # joint response is genuinely sub-additive at high forcing.
        self.interaction_loss_scaling      = getattr(self, "interaction_loss_scaling", 0.0)
        self._interaction_target_fraction  = getattr(self, "interaction_target_fraction", 0.05)

        # ── Global-mean supervision (per-sample, fixed scaling, all scenarios) ──
        self.gmean_loss_scaling = getattr(self, "gmean_loss_scaling", 0.0)

        # ── Energy-balance constraint scaling and target ──
        # The scalar parameters themselves are registered on the model in
        # `_register_ebm_scalars()` (DDP-syncable + saved with model state).
        self.ebm_loss_scaling     = getattr(self, "ebm_loss_scaling",     0.0)
        self._ebm_target_fraction = getattr(self, "ebm_target_fraction",  0.05)

    def _register_ebm_scalars(self) -> None:
        """Create the three EBM scalars (α_ghg, α_aero, λ) and attach them to
        `self.model` so DDP syncs their gradients and they're persisted in the
        model state_dict. Must run BEFORE the optimizer is constructed.
        """
        self._ebm_alpha_ghg  = torch.nn.Parameter(torch.tensor( 1.0))
        self._ebm_alpha_aero = torch.nn.Parameter(torch.tensor(-1.0))
        self._ebm_lambda     = torch.nn.Parameter(torch.tensor(-1.0))
        self.model.register_parameter("ebm_alpha_ghg",  self._ebm_alpha_ghg)
        self.model.register_parameter("ebm_alpha_aero", self._ebm_alpha_aero)
        self.model.register_parameter("ebm_lambda",     self._ebm_lambda)

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
        epoch_anom_signals = []
        epoch_anom_errors = []
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

                loss, mse_loss, cond_loss, anom_signal, anom_error, sens, scen_disc, tcre_loss, ebm_loss = self.get_loss(batch, cond, scenario_ids=scenario_ids)

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
                    avg_ebm_loss    = self.accelerator.gather_for_metrics(ebm_loss).mean()
                    avg_anom_error  = self.accelerator.gather_for_metrics(anom_error).mean()
                    avg_anom_signal = self.accelerator.gather_for_metrics(anom_signal).mean()
                    avg_sens        = self.accelerator.gather_for_metrics(sens).mean()
                    avg_scen_disc   = self.accelerator.gather_for_metrics(scen_disc).mean()

                    log_dict = {
                        "Training/Loss":  avg_loss.detach().item(),
                        "MSE LOSS":       avg_mse_loss.detach().item(),
                        "COND LOSS":      avg_cond_loss.detach().item(),
                        "TCRE LOSS":      avg_tcre_loss.detach().item(),
                        "EBM LOSS":       avg_ebm_loss.detach().item(),
                        "INTER LOSS":     float(getattr(self, "_cached_interaction", 0.0)),
                        "GMEAN LOSS":     float(getattr(self, "_cached_gmean", 0.0)),
                        "NULL DRIFT":     float(getattr(self, "_cached_null_drift", 0.0)),
                        "ANOM ERROR":     avg_anom_error.detach().item(),
                        "ANOM SIGNAL":    avg_anom_signal.detach().item(),
                        "SENS":           avg_sens.detach().item(),
                        "ANOM SKILL":     (1.0 - avg_anom_error / (avg_anom_signal + 1e-6)).item(),
                        "SCEN DISC":      avg_scen_disc.detach().item(),
                        "COND SCALE":     self.cond_loss_scaling,
                        "TCRE SCALE":     self.tcre_loss_scaling,
                        "TCRE SLOPE":     (self.tcre_slope if self.tcre_slope is not None else 0.0),
                        "EBM SCALE":      self.ebm_loss_scaling,
                        "INTER SCALE":    self.interaction_loss_scaling,
                        "GMEAN SCALE":    self.gmean_loss_scaling,
                        "EBM/alpha_ghg":  float(self._ebm_alpha_ghg.detach().item()),
                        "EBM/alpha_aero": float(self._ebm_alpha_aero.detach().item()),
                        "EBM/lambda":     float(self._ebm_lambda.detach().item()),
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

        if self.cond_loss_scaling > 0 and self.climatology is not None:
            null_cond   = torch.full_like(cond_map, -1.0)
            null_output = ema_model(noisy_samples, timesteps, cond_map=null_cond)

            pred_x0_cond = self.get_original_sample(noisy_samples, model_output, timesteps)
            pred_x0_null = self.get_original_sample(noisy_samples, null_output, timesteps)

            baseline     = self.climatology.to(device=clean_samples.device, dtype=clean_samples.dtype)
            true_anomaly = self._scenario_ensemble_mean(clean_samples - baseline, scenario_ids)
            pred_anomaly = pred_x0_cond - pred_x0_null

            cond_loss   = calc_mse_loss(pred_anomaly, true_anomaly, self._ref_ds.lats)
            anom_signal = true_anomaly.abs().mean()
            anom_error  = (pred_anomaly - true_anomaly).abs().mean()

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
            anom_signal = torch.zeros(1, device=self.device)
            anom_error  = torch.zeros(1, device=self.device)
            scen_disc   = torch.zeros(1, device=self.device)

        return mse, cond_loss, anom_signal, anom_error, scen_disc

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

            mse, cond_l, sig, err, disc = self._compute_val_metrics(batch, cond, scenario_ids)

            accum["mse"].append(self.accelerator.gather_for_metrics(mse).mean().item())
            accum["cond"].append(self.accelerator.gather_for_metrics(cond_l).mean().item())
            accum["signal"].append(self.accelerator.gather_for_metrics(sig).mean().item())
            accum["error"].append(self.accelerator.gather_for_metrics(err).mean().item())
            accum["disc"].append(self.accelerator.gather_for_metrics(disc).mean().item())

        import numpy as _np
        avg_sig   = _np.mean(accum["signal"])
        avg_err   = _np.mean(accum["error"])
        val_skill = float(1.0 - avg_err / (avg_sig + 1e-6))

        log_dict = {
            "VAL/MSE":         _np.mean(accum["mse"]),
            "VAL/Skill":       val_skill,
            "VAL/DISC":        _np.mean(accum["disc"]),
            "VAL/ANOM_ERROR":  avg_err,
            "VAL/ANOM_SIGNAL": avg_sig,
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
        except Exception as e:
            self.accelerator.print(
                f"  [EVAL] WARNING: could not write eval trigger for epoch {epoch}: {e}"
            )

    def _update_loss_emas(self) -> None:
        """Update exponential moving averages of unscaled mse, cond, tcre, ebm loss."""
        d = self._ema_decay
        mse_val  = getattr(self, "_last_raw_mse",  0.0)
        cond_val = getattr(self, "_last_raw_cond", 0.0)
        tcre_val = getattr(self, "_last_raw_tcre", 0.0)
        ebm_val  = getattr(self, "_last_raw_ebm",  0.0)
        inter_val = getattr(self, "_last_raw_interaction", 0.0)
        gmean_val = getattr(self, "_last_raw_gmean", 0.0)
        self._ema_mse = mse_val if self._ema_mse is None else d * self._ema_mse + (1 - d) * mse_val
        if cond_val > 1e-8:
            self._ema_cond = cond_val if self._ema_cond is None else d * self._ema_cond + (1 - d) * cond_val
        if tcre_val > 1e-8:
            self._ema_tcre = tcre_val if self._ema_tcre is None else d * self._ema_tcre + (1 - d) * tcre_val
        if ebm_val > 1e-8:
            self._ema_ebm = ebm_val if self._ema_ebm is None else d * self._ema_ebm + (1 - d) * ebm_val
        if inter_val > 1e-8:
            self._ema_interaction = inter_val if self._ema_interaction is None else d * self._ema_interaction + (1 - d) * inter_val
        if gmean_val > 1e-8:
            self._ema_gmean = gmean_val if self._ema_gmean is None else d * self._ema_gmean + (1 - d) * gmean_val

    def _update_cond_scaling(self, sensitivity_value: float, epoch: int) -> None:
        """Update cond_loss_scaling on a fixed epoch-based schedule.

        Two phases:
          1. Warmup  (epoch < cond_warmup_epochs):
               scaling = 0.0 — let MSE dominate, build a stable baseline.
          2. Linear ramp then hold  (epoch >= cond_warmup_epochs):
               scaling ramps linearly 0 → cond_max_scaling over cond_ramp_epochs,
               then holds at cond_max_scaling.

        If adaptive_loss_scaling=True, tcre/ebm/cond scalings are nudged toward
        their target_fraction of MSE each sync step using EMAs of the raw losses.
        """
        # ── MSE-only mode: hold cond_loss_scaling=0 so the whole _sync_with_cond
        # block (null pass + cond/TCRE/low-t/EBM/interaction) is skipped → pure
        # denoising MSE. Diagnostic arm (does balanced sampling let plain MSE
        # learn the right sensitivity without the aux losses?).
        if getattr(self, "mse_only", False):
            self.cond_loss_scaling = 0.0
            return

        # ── Phase 1: warmup ───────────────────────────────────────────────────
        if epoch < self.cond_warmup_epochs:
            self.cond_loss_scaling = 0.0
            return

        # ── Phase 2: linear ramp then hold ────────────────────────────────────
        ramp_epoch = epoch - self.cond_warmup_epochs
        progress = min(ramp_epoch / self.cond_ramp_epochs, 1.0)  # clamp at 1.0 after ramp
        self.cond_loss_scaling = self.cond_max_scaling * progress

        # ── Adaptive scalings ─────────────────────────────────────────────────
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

        # tcre_loss: free range [0.001, 0.5]
        if self._ema_tcre is not None and self._ema_tcre > 1e-8:
            target = self._tcre_target_fraction * self._ema_mse
            self.tcre_loss_scaling = float(max(0.001, min(0.5, target / self._ema_tcre)))

        # ebm_loss: free range [0.001, 0.5] (only when explicitly enabled).
        # EBM is DROPPED by default (ebm_loss_scaling: 0 in config); this branch
        # is inert unless re-enabled. See memory ebm_term_near_inactive.
        if self.ebm_loss_scaling > 0 and self._ema_ebm is not None and self._ema_ebm > 1e-8:
            target = self._ebm_target_fraction * self._ema_mse
            self.ebm_loss_scaling = float(max(0.001, min(0.5, target / self._ema_ebm)))

        # interaction_loss: free range [0.001, 0.5] (only when explicitly enabled)
        if (self.interaction_loss_scaling > 0 and self._ema_interaction is not None
                and self._ema_interaction > 1e-8):
            target = self._interaction_target_fraction * self._ema_mse
            self.interaction_loss_scaling = float(max(0.001, min(0.5, target / self._ema_interaction)))

    def _precompute_tcre_slope(self) -> None:
        """Fit slope+intercept of gmean(ΔT_norm) vs gmean(cumCO2_norm) on
        the concatenated hist + ssp370 trajectory.

        This matches the all-forcings TCRE reference reported by eval
        (`replot_eval.py`, TCRE_SCENARIOS={"hist","ssp370"}).  The slope
        therefore includes aerosol cooling and is correctly compared
        against the model's *full* predicted anomaly on hist+ssp370 samples.

        Runs on the main process only.  On failure (missing scenarios,
        degenerate range), disables the TCRE loss rather than raising.
        """
        if not self._multi:
            print("[TRAINER] _precompute_tcre_slope: single-experiment mode — skipping TCRE")
            self.tcre_loss_scaling = 0.0
            return

        scenario_names = self.train_set.scenario_names
        for required in ("hist", "ssp370"):
            if required not in scenario_names:
                print(f"[TRAINER] _precompute_tcre_slope: '{required}' missing from "
                      f"{scenario_names} — skipping TCRE")
                self.tcre_loss_scaling = 0.0
                return

        # Build area weights (cosine-latitude, clamped — same as training loss)
        lats = torch.as_tensor(self._ref_ds.lats.values, dtype=torch.float32)
        w_lat = torch.cos(torch.deg2rad(lats)).clamp(min=0.2)
        w_lat = w_lat / w_lat.mean()
        w_lat = w_lat.view(1, -1, 1)  # (1, H, 1)

        # Climatology baseline (1850-1900 mean) — anchors the ΔT axis.
        if self.climatology is not None:
            clim = self.climatology.detach().to("cpu").squeeze(0).squeeze(1)  # (C, H, W)
        else:
            clim = None

        def _gmean_trajectory(scen_name):
            ds = self.train_set.datasets[scenario_names.index(scen_name)]
            if ds.tensor_data is None or ds.tensor_data_cond is None:
                first_real = list(ds.realizations)[0]
                print(f"[TRAINER] _precompute_tcre_slope: loading {scen_name} "
                      f"realization '{first_real}'")
                ds.load_data(first_real)
            t = ds.tensor_data           # (n_vars_target, T, H, W)
            c = ds.tensor_data_cond      # (n_vars_cond,   T, H, W) — ch 0 = CO2
            clim_t = clim[0] if clim is not None else torch.zeros(t.shape[2], t.shape[3])
            dT  = t[0] - clim_t.unsqueeze(0)           # (T, H, W)
            co2 = c[0]                                  # (T, H, W)
            dT_gm  = (dT  * w_lat).mean(dim=(1, 2))    # (T,)
            co2_gm = (co2 * w_lat).mean(dim=(1, 2))    # (T,)
            return co2_gm.to(torch.float64).numpy(), dT_gm.to(torch.float64).numpy(), co2

        try:
            x_h, y_h, co2_h = _gmean_trajectory("hist")
            x_s, y_s, co2_s = _gmean_trajectory("ssp370")
        except Exception as e:
            print(f"[TRAINER] _precompute_tcre_slope: failed to load hist/ssp370: {e}")
            self.tcre_loss_scaling = 0.0
            return

        import numpy as _np
        x = _np.concatenate([x_h, x_s])
        y = _np.concatenate([y_h, y_s])

        print(f"[TRAINER] _precompute_tcre_slope: hist T={len(x_h)} "
              f"area-w gmean=[{x_h.min():.4g}, {x_h.max():.4g}]  "
              f"ssp370 T={len(x_s)} area-w gmean=[{x_s.min():.4g}, {x_s.max():.4g}]")

        if x.size < 3 or float(x.max() - x.min()) < 1e-6:
            print("[TRAINER] _precompute_tcre_slope: degenerate CO2 range — skipping TCRE")
            self.tcre_loss_scaling = 0.0
            return

        slope, intercept = _np.polyfit(x, y, 1)
        self.tcre_slope     = float(slope)
        self.tcre_intercept = float(intercept)
        print(
            f"[TRAINER] TCRE precompute (hist+ssp370 combined): slope={self.tcre_slope:.4f} "
            f"intercept={self.tcre_intercept:.4f}  "
            f"(CO2_norm: [{x.min():.3f}, {x.max():.3f}], "
            f"ΔT_norm: [{y.min():.3f}, {y.max():.3f}])"
        )

        # ── Per-scenario targets (#1 local slope-match) ──────────────────────
        # Fit hist and ssp370 separately so the loss pins each scenario's
        # ΔT-vs-cumCO2 slope to its own CESM2 value.  scenario_ids: hist=0,
        # ssp370=1.  Fall back to the combined line for any scenario whose
        # CO2 range is degenerate.
        self.tcre_slopes = {}
        for sid, (xs_, ys_, nm) in {0: (x_h, y_h, "hist"),
                                    1: (x_s, y_s, "ssp370")}.items():
            if len(xs_) >= 2 and float(xs_.max() - xs_.min()) >= 1e-6:
                m_, b_ = _np.polyfit(xs_, ys_, 1)
                self.tcre_slopes[sid] = (float(m_), float(b_))
            else:
                self.tcre_slopes[sid] = (self.tcre_slope, self.tcre_intercept)
            print(f"[TRAINER] TCRE precompute ({nm}): "
                  f"slope={self.tcre_slopes[sid][0]:.4f} "
                  f"intercept={self.tcre_slopes[sid][1]:.4f}")

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

    # ─────────────────────────────────────────────────────────────────────────
    # Aux-loss components used by get_loss.
    # Each returns a scalar tensor (or a (1,)-tensor of zeros if disabled), so
    # the caller can multiply by its scaling factor and add to the total loss
    # without further branching.
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_ebm_loss(self, pred_anomaly: torch.Tensor,
                          cond_map: torch.Tensor) -> torch.Tensor:
        """Energy-balance constraint: (F_ghg + F_aero + λ·ΔT)² per-sample, mean.

        F_ghg = α_ghg · gmean(cumCO2_norm),  F_aero = α_aero · gmean(SUL_norm).
        Returns zeros when ebm_loss_scaling <= 0.
        """
        if self.ebm_loss_scaling <= 0:
            return torch.zeros(1, device=self.device)
        w_be = _area_weights_1d(
            self._ref_ds.lats, dtype=pred_anomaly.dtype, device=pred_anomaly.device,
        ).view(1, 1, 1, -1, 1)
        dT_gmean  = (pred_anomaly         * w_be).mean(dim=(1, 2, 3, 4))
        co2_gmean = (cond_map[:, 0:1]     * w_be).mean(dim=(1, 2, 3, 4))
        sul_gmean = (cond_map[:, 1:2]     * w_be).mean(dim=(1, 2, 3, 4))
        F_ghg    = self._ebm_alpha_ghg  * co2_gmean
        F_aero   = self._ebm_alpha_aero * sul_gmean
        residual = F_ghg + F_aero + self._ebm_lambda * dT_gmean
        return (residual ** 2).mean().unsqueeze(0)

    def _compute_tcre_loss(self, pred_x0_cond: torch.Tensor,
                           pred_anomaly: torch.Tensor,
                           cond_map: torch.Tensor,
                           scenario_ids,
                           pred_x0_lowt: torch.Tensor = None) -> torch.Tensor:
        """Per-scenario slope-match: ΔT_norm ≈ slope · cumCO2_norm + intercept.

        Slopes precomputed from CESM2 hist + ssp370 trajectories
        (`_precompute_tcre_slope`). Only scenarios present in self.tcre_slopes
        contribute (hist=0, ssp370=1 by default).

        pred_x0_lowt: when provided (tcre_lowt path), the constraint is scored on
        this LOW-noise x0 prediction as a full anomaly (pred − climatology), where
        one-step x0 ≈ the true sample — so it pins the *sampled* sensitivity rather
        than the random-t one-step proxy.
        """
        if not (self.tcre_loss_scaling > 0 and self.tcre_slopes and scenario_ids is not None):
            return torch.zeros(1, device=self.device)

        # Score the same quantity eval reports (full anomaly = pred − climatology)
        # when on the low-t path or tcre_full_anomaly; else the CFG decomposition.
        if pred_x0_lowt is not None:
            tcre_pred = pred_x0_lowt - self.climatology.to(dtype=pred_x0_lowt.dtype)
        elif self.tcre_full_anomaly:
            tcre_pred = pred_x0_cond - self.climatology.to(dtype=pred_x0_cond.dtype)
        else:
            tcre_pred = pred_anomaly

        w_b = _area_weights_1d(
            self._ref_ds.lats, dtype=pred_anomaly.dtype, device=pred_anomaly.device,
        ).view(1, 1, 1, -1, 1)

        tcre_terms = []
        for sid, (slope, intercept) in self.tcre_slopes.items():
            m_sid = (scenario_ids == sid)
            if m_sid.sum() == 0:
                continue
            dT_gmean  = (tcre_pred[m_sid]          * w_b).mean(dim=(1, 2, 3, 4))
            co2_gmean = (cond_map[m_sid][:, 0:1]   * w_b).mean(dim=(1, 2, 3, 4))
            target_dT = slope * co2_gmean + intercept
            tcre_terms.append((dT_gmean - target_dT) ** 2)
        if not tcre_terms:
            return torch.zeros(1, device=self.device)
        return torch.cat(tcre_terms).mean()

    def _compute_interaction_loss(self, pred_x0_cond: torch.Tensor,
                                  pred_x0_null: torch.Tensor,
                                  cond_map_input: torch.Tensor,
                                  noisy_samples: torch.Tensor,
                                  timesteps: torch.Tensor,
                                  scenario_ids) -> torch.Tensor:
        """Additivity penalty: interaction = f(CO2,SUL) − f(CO2,0) − f(0,SUL) + f(0,0).

        Applied to hist samples only (low-forcing). Two extra NO-GRAD marginal
        forward passes; gradient flows only through pred_x0_cond[lf_mask].
        """
        if not (self.interaction_loss_scaling > 0 and scenario_ids is not None):
            return torch.zeros(1, device=self.device)
        lf_mask = (scenario_ids == 0)                 # hist only
        if lf_mask.sum() == 0:
            return torch.zeros(1, device=self.device)

        ci = cond_map_input[lf_mask]
        ns = noisy_samples[lf_mask]
        ts = timesteps[lf_mask]
        co2only = ci.clone(); co2only[:, 1] = NULL_COND_VALUE       # SUL→null
        sulonly = ci.clone(); sulonly[:, 0] = NULL_COND_VALUE       # CO2→null
        with torch.no_grad():
            p_co2 = self.get_original_sample(ns, self.model(ns, ts, cond_map=co2only), ts)
            p_sul = self.get_original_sample(ns, self.model(ns, ts, cond_map=sulonly), ts)
        interaction = (
            pred_x0_cond[lf_mask]
            - p_co2.detach() - p_sul.detach()
            + pred_x0_null[lf_mask].detach()
        )
        return (interaction ** 2).mean().unsqueeze(0)

    def _compute_gmean_loss(self, pred_x0_cond: torch.Tensor,
                            target_gm: torch.Tensor,
                            cond_dropped: torch.Tensor) -> torch.Tensor:
        """Pin the predicted field's area-weighted global mean to the target's.

        Excludes CFG-dropped samples (their conditioning was nulled, so their
        prediction shouldn't be matched to the fully-forced target).
        """
        if self.gmean_loss_scaling <= 0 or not (~cond_dropped).any():
            return torch.zeros(1, device=self.device)
        intact = ~cond_dropped
        w_g = _area_weights_1d(
            self._ref_ds.lats, dtype=pred_x0_cond.dtype, device=pred_x0_cond.device,
        ).view(1, 1, 1, -1, 1)
        pred_gm = (pred_x0_cond * w_g).mean(dim=(1, 2, 3, 4))
        return ((pred_gm[intact] - target_gm[intact]) ** 2).mean().unsqueeze(0)

    def _log_null_drift(self, pred_x0_null: torch.Tensor) -> None:
        """Diagnostic: gmean(pred_x0_null − climatology) — should be ~0 if calibrated.

        Side effect: caches the value in self._cached_null_drift for the next log entry.
        """
        with torch.no_grad():
            w_n = _area_weights_1d(
                self._ref_ds.lats, dtype=pred_x0_null.dtype, device=pred_x0_null.device,
            ).view(1, 1, 1, -1, 1)
            drift = ((pred_x0_null - self.climatology.to(dtype=pred_x0_null.dtype)) * w_n).mean()
            self._cached_null_drift = drift.item()

    def _compute_scen_disc(self, model_output: torch.Tensor,
                           noisy_samples: torch.Tensor,
                           timesteps: torch.Tensor,
                           scenario_ids) -> torch.Tensor:
        """Mean pairwise L1 distance between per-scenario mean x0 predictions."""
        if scenario_ids is None:
            return torch.zeros(1, device=self.device)
        unique_scenarios = scenario_ids.unique()
        if len(unique_scenarios) < 2:
            return torch.zeros(1, device=self.device)
        # Use model_output detached so this metric doesn't pull gradients.
        pred_x0_scen = self.get_original_sample(noisy_samples, model_output.detach(), timesteps)
        scenario_means = []
        for sid in unique_scenarios:
            mask_s = scenario_ids == sid
            if mask_s.sum() > 0:
                scenario_means.append(pred_x0_scen[mask_s].mean(dim=0))
        if len(scenario_means) < 2:
            return torch.zeros(1, device=self.device)
        n = len(scenario_means)
        pair_dists = [
            (scenario_means[i] - scenario_means[j]).abs().mean()
            for i in range(n) for j in range(i + 1, n)
        ]
        return torch.stack(pair_dists).mean()

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

            gmean_loss = torch.zeros(1, device=self.device)  # global-mean supervision (set below if enabled)

            # ── Per-channel CFG dropout ───────────────────────────────────────
            # Independently zero CO2 (ch 0) and SUL (ch 1) for randomly chosen
            # batch elements. This trains the model on all four conditioning
            # subsets so that per-channel guidance works correctly at inference.
            # Only applied to scenarios that vary both channels (hist, ssp370):
            # aaer has constant CO2 and ghg has constant SUL, so dropping the
            # constant channel teaches the model nothing useful and dropping
            # the varying channel collapses the only signal those samples carry.
            cond_dropped = torch.zeros(clean_samples.shape[0], device=self.device, dtype=torch.bool)
            if self.cfg_co2_drop_prob > 0 or self.cfg_sul_drop_prob > 0:
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
                # Stash the target's area-weighted global mean for gmean_loss
                # BEFORE clean_samples is freed (the loss is computed later, after
                # the forward passes, by which point clean_samples is gone).
                # Default to None so the (unconditional) _compute_gmean_loss call
                # site always has a bound argument when gmean_loss is disabled.
                target_gm = None
                if self.gmean_loss_scaling > 0:
                    _wt = torch.cos(torch.deg2rad(torch.as_tensor(
                        self._ref_ds.lats.values,
                        dtype=clean_samples.dtype, device=clean_samples.device,
                    ))).clamp(min=0.2)
                    _wt = (_wt / _wt.mean()).view(1, 1, 1, -1, 1)
                    target_gm = (clean_samples * _wt).mean(dim=(1, 2, 3, 4)).detach()
            else:
                true_anomaly = None

            # ── Low-t TCRE: build a fixed low-noise input before clean_samples is
            # freed. The forward pass runs later (in the sync block) so its grad
            # graph only lives on the sync iteration. (tcre_lowt path only.)
            tcre_lowt_active = (
                _sync_with_cond and self.tcre_lowt and self.tcre_loss_scaling > 0
            )
            if tcre_lowt_active:
                t_low = self.scheduler.log_snr(torch.full(
                    (clean_samples.shape[0],), self.tcre_lowt_level,
                    device=clean_samples.device,
                ))
                noisy_low = self.scheduler.add_noise(
                    clean_samples, torch.randn_like(clean_samples), t_low,
                )
            else:
                t_low = noisy_low = None

            del clean_samples  # no longer needed; free before forward passes

            # ── Conditioned forward pass (with gradients) ─────────────────────
            model_output = self.model(noisy_samples, timesteps, cond_map=cond_map_input)

            # ── Primary denoising loss ────────────────────────────────────────
            mse_loss = calc_mse_loss(model_output, target, self._ref_ds.lats)

            # DDP sentinel: keep the EBM scalars in the autograd graph on every
            # iteration so DDP's reducer always sees a grad path for them.
            # Without this, accumulation iters that skip the aux branch leave
            # ebm_alpha_ghg/ebm_alpha_aero/ebm_lambda with no gradient → crash
            # on the next forward (`parameters which did not receive grad`).
            mse_loss = mse_loss + 0.0 * (
                self._ebm_alpha_ghg + self._ebm_alpha_aero + self._ebm_lambda
            )

            # Only compute the expensive null forward pass on the final accumulation
            # step (when gradients are about to sync).  With gradient_accumulation_steps=4
            # this cuts null-pass frequency from 4× to 1× per optimizer step — a 4×
            # reduction in null-pass overhead — with no loss in gradient quality
            # (gradients only backprop through the conditioned pass anyway).
            if _sync_with_cond:
                # ── Null forward pass (no gradients — metric + baseline only) ─
                # Run the same noisy batch through the model with all-null
                # conditioning (-1.0 = pre-industrial baseline under normalisation).
                # No grad needed: this pass is reference-only; gradients flow only
                # through the conditioned pass above.
                with torch.no_grad():
                    null_cond = torch.full_like(cond_map, NULL_COND_VALUE)
                    null_output = self.model(noisy_samples, timesteps, cond_map=null_cond)
                    del null_cond

                # ── Decode both outputs to x0 space ──────────────────────────
                pred_x0_cond = self.get_original_sample(noisy_samples, model_output, timesteps)
                with torch.no_grad():
                    pred_x0_null = self.get_original_sample(noisy_samples, null_output, timesteps)
                del null_output  # no longer needed; not part of grad graph

                # ── Sensitivity metric: how much does conditioning move output ─
                # Measured in x0 space so it's in physical units (not noise space).
                cond_sensitivity = (pred_x0_cond - pred_x0_null).abs().mean().detach()
                self._cached_sensitivity = cond_sensitivity.item()

                # ── Predicted anomaly: cond output - null output ──────────────
                # The difference between the conditioned and null model x0
                # predictions is exactly what the emission forcing added.
                # Gradients flow only through pred_x0_cond (null is no_grad).
                pred_anomaly = pred_x0_cond - pred_x0_null.detach()

                # ── Aux losses (each helper returns zeros when its scaling is 0) ──
                ebm_loss         = self._compute_ebm_loss(pred_anomaly, cond_map)
                # Low-t TCRE: extra conditioned forward at fixed low noise, where
                # one-step x0 ≈ the true sample → constrains the *sampled*
                # sensitivity (escapes the random-t one-step proxy that lets the
                # ~+13% overshoot survive). Same conditioning as the main pass.
                pred_x0_lowt = None
                if tcre_lowt_active:
                    out_low = self.model(noisy_low, t_low, cond_map=cond_map_input)
                    pred_x0_lowt = self.get_original_sample(noisy_low, out_low, t_low)
                    del out_low, noisy_low
                tcre_loss        = self._compute_tcre_loss(
                    pred_x0_cond, pred_anomaly, cond_map, scenario_ids,
                    pred_x0_lowt=pred_x0_lowt,
                )
                interaction_loss = self._compute_interaction_loss(
                    pred_x0_cond, pred_x0_null, cond_map_input,
                    noisy_samples, timesteps, scenario_ids,
                )
                gmean_loss       = self._compute_gmean_loss(
                    pred_x0_cond, target_gm, cond_dropped,
                )

                # Diagnostic only — caches self._cached_null_drift, no grad.
                self._log_null_drift(pred_x0_null)

                del pred_x0_cond, pred_x0_null

                # ── Conditioning loss: direct anomaly regression ──────────────
                cond_loss = calc_mse_loss(pred_anomaly, true_anomaly, self._ref_ds.lats)
                anom_signal = true_anomaly.abs().mean().detach()
                anom_error  = (pred_anomaly - true_anomaly).abs().mean().detach()
                del true_anomaly, pred_anomaly

                # ── Scenario discrimination metric (per-scenario mean x0 spread) ─
                scen_disc = self._compute_scen_disc(
                    model_output, noisy_samples, timesteps, scenario_ids,
                )

            elif self.cond_loss_scaling > 0:
                # cond_loss_scaling is active but this is a non-sync accumulation step —
                # skip the null pass entirely.  Use cached sensitivity from last sync step.
                cond_loss        = torch.zeros(1, device=self.device)
                tcre_loss        = torch.zeros(1, device=self.device)
                ebm_loss         = torch.zeros(1, device=self.device)
                interaction_loss = torch.zeros(1, device=self.device)
                anom_error       = torch.zeros(1, device=self.device)
                anom_signal      = torch.zeros(1, device=self.device)
                cond_sensitivity = torch.tensor(self._cached_sensitivity, device=self.device)
                scen_disc        = torch.zeros(1, device=self.device)

            else:
                cond_loss        = torch.zeros(1, device=self.device)
                tcre_loss        = torch.zeros(1, device=self.device)
                ebm_loss         = torch.zeros(1, device=self.device)
                interaction_loss = torch.zeros(1, device=self.device)
                anom_error       = torch.zeros(1, device=self.device)
                anom_signal      = torch.zeros(1, device=self.device)
                cond_sensitivity = torch.zeros(1, device=self.device)
                scen_disc        = torch.zeros(1, device=self.device)

            # Cache raw unscaled loss values for adaptive scaling EMA (before multiply)
            self._last_raw_mse  = mse_loss.detach().item()
            self._last_raw_cond = cond_loss.detach().item()
            self._last_raw_tcre = tcre_loss.detach().item()
            self._last_raw_ebm  = ebm_loss.detach().item()
            self._last_raw_interaction = interaction_loss.detach().item()
            self._cached_interaction   = self._last_raw_interaction
            self._last_raw_gmean = gmean_loss.detach().item()
            self._cached_gmean   = self._last_raw_gmean

            # ── Total loss ────────────────────────────────────────────────────
            loss = (
                mse_loss
                + cond_loss * self.cond_loss_scaling
                + tcre_loss * self.tcre_loss_scaling
                + ebm_loss  * self.ebm_loss_scaling
                + interaction_loss * self.interaction_loss_scaling
                + gmean_loss * self.gmean_loss_scaling
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
            anom_signal, anom_error, cond_sensitivity, scen_disc,
            tcre_loss * self.tcre_loss_scaling,
            ebm_loss  * self.ebm_loss_scaling,
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
        }
        for ckpt_key, attr, _ in self._PERSISTED_FIELDS:
            sd[ckpt_key] = getattr(self, attr)
        if extra:
            sd.update(extra)
        return sd

    def _restore_persisted_fields(self, checkpoint: dict) -> None:
        """Restore aux-loss scalings / EMAs from a checkpoint dict.

        Honours the `zero_disables` flag: a restored 0 (or None) for a field
        marked zero_disables=True keeps the current (config-provided) value.
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

        # ── Rotate: keep the 5 newest numbered checkpoints (best.pt is excluded) ─
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
        for stale in sorted(existing, key=_epoch_from_name, reverse=True)[5:]:
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
        # ebm_alpha_ghg / ebm_alpha_aero / ebm_lambda will keep their inits.
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

        # Restore aux-loss scalings / EMAs / tcre slope from the checkpoint.
        # Fields marked zero_disables=True in _PERSISTED_FIELDS keep the
        # config-provided value when the checkpoint stored a 0, so a previously
        # disabled-by-bug run doesn't sticky-disable the loss on resume.
        self._restore_persisted_fields(checkpoint)
        print(f"[INFO] Restored cond_loss_scaling={self.cond_loss_scaling:.4f}  "
              f"tcre_loss_scaling={self.tcre_loss_scaling:.4f}  "
              f"tcre_slope={self.tcre_slope}")

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