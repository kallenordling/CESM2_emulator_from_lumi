import lumi_paths as L
import os
# Must be set before any GPU allocation (before torch import)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import DDPMScheduler

from data.multi_experiment_dataset import MultiExperimentDataset, build_multi_experiment_loader
from trainer.unetTrainer import UNetTrainer
from models.video_net import UNetModel3D
import torch
print(f"PyTorch: {torch.__version__}  ROCm/HIP: {torch.version.hip}  CUDA: {torch.version.cuda}")
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed')
warnings.filterwarnings('ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config_aero.yaml")
def main(cfg: DictConfig) -> None:

    # Default DDP: every parameter (including EBM scalars) receives a gradient
    # on every backward (the aux branch runs every iter, see unetTrainer.py:912),
    # so neither find_unused_parameters nor static_graph is needed.
    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.mixed_precision,
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        split_batches=cfg.accelerator.get('split_batches', False),
    )

    set_seed(cfg.seed, device_specific=False)
    logger = get_logger(__name__, log_level="INFO")

    # ── DDP verification: every rank prints its identity ─────────────────
    # Confirms (a) num_processes == expected GPU count, (b) each rank is
    # bound to a distinct device, (c) HOSTNAME differs across nodes.
    import socket
    print(
        f"[DDP] rank={accelerator.process_index}/{accelerator.num_processes} "
        f"local_rank={accelerator.local_process_index} "
        f"device={accelerator.device} "
        f"host={socket.gethostname()} "
        f"pid={os.getpid()}",
        flush=True,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[DDP] all {accelerator.num_processes} ranks reported in", flush=True)

    # ── Load data config directly from its own yaml file ───────────────────
    # The data config is flat (no `data:` wrapper) so we load it separately
    # rather than composing it into config_aero.yaml via defaults.
    data_cfg_path = os.path.join(
        hydra.utils.get_original_cwd(), "configs", cfg.data_config
    )
    data_cfg = L.resolve_cfg(OmegaConf.load(data_cfg_path))

    # BC clip mode must be set BEFORE any dataset/cond building (it changes the
    # percentile anchors of the BC channel normalisation). Default "v1" keeps
    # existing runs byte-identical; the actual (lo, hi) used is persisted into
    # checkpoints (COND_NORM) so eval stays consistent per checkpoint.
    # Read order: trainer.hyperparameters.bc_clip_mode when non-null (CLI
    # override, e.g. the run2_gainfix.sh fork; the config_aero default is null
    # precisely so this source stays empty unless explicitly set) → data
    # config key → "v1".
    bc_clip_mode = str(
        OmegaConf.select(cfg, "trainer.hyperparameters.bc_clip_mode")
        or data_cfg.get("bc_clip_mode", None)
        or "v1"
    )
    if bc_clip_mode != "v1":
        from data import climate_dataset as _cds
        _cds.set_bc_clip_mode(bc_clip_mode)
        if accelerator.is_main_process:
            logger.info(f"BC clip mode: {bc_clip_mode}")

    if accelerator.is_main_process:
        logger.info(f"Rank {accelerator.process_index}/{accelerator.num_processes}")
        logger.info(f"Loaded data config from: {data_cfg_path}")
        logger.info("Building multi-experiment dataset...")

    # ── Build multi-experiment train dataset ────────────────────────────────
    train_set = build_multi_experiment_loader(
        experiment_configs=OmegaConf.to_container(data_cfg.experiment_configs, resolve=True),
        accelerator=accelerator,
        batch_size=data_cfg.batch_size,
        mix_scenarios=data_cfg.get("mix_scenarios", True),
        steps_per_realization=data_cfg.get("steps_per_realization", None),
        scenario_weights=OmegaConf.to_container(data_cfg.scenario_weights, resolve=True) if data_cfg.get("scenario_weights") is not None else None,
        year_bias=data_cfg.get("year_bias", 0.0),
        year_bias_floor=data_cfg.get("year_bias_floor", 0.05),
        bsp_depth=data_cfg.get("bsp_depth", 0),
        # shared ClimateDataset kwargs
        seq_len=data_cfg.seq_len,
        target_vars=OmegaConf.to_container(data_cfg.target_vars, resolve=True),
        cond_vars=OmegaConf.to_container(data_cfg.cond_vars, resolve=True),
        n_components_target=data_cfg.get("n_components_target", None),
        n_components_cond=data_cfg.get("n_components_cond", None),
        cond_smooth_sigma=data_cfg.get("cond_smooth_sigma", None),
        cond_smooth_method=data_cfg.get("cond_smooth_method", "gaussian"),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── Build held-out validation dataset ───────────────────────────────────
    val_set = None
    if data_cfg.get("val_experiment_configs") is not None:
        val_set = build_multi_experiment_loader(
            experiment_configs=OmegaConf.to_container(data_cfg.val_experiment_configs, resolve=True),
            accelerator=accelerator,
            batch_size=data_cfg.batch_size,
            mix_scenarios=data_cfg.get("mix_scenarios", True),
            steps_per_realization=data_cfg.get("steps_per_realization", None),
            seq_len=data_cfg.seq_len,
            target_vars=OmegaConf.to_container(data_cfg.target_vars, resolve=True),
            cond_vars=OmegaConf.to_container(data_cfg.cond_vars, resolve=True),
            n_components_target=data_cfg.get("n_components_target", None),
            n_components_cond=data_cfg.get("n_components_cond", None),
        cond_smooth_sigma=data_cfg.get("cond_smooth_sigma", None),
        cond_smooth_method=data_cfg.get("cond_smooth_method", "gaussian"),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            shard_across_ranks=False,  # every rank evaluates the full val set
        )
        if accelerator.is_main_process:
            logger.info(f"Validation set: {val_set.dataset.scenario_names} (1 held-out member each)")

    if accelerator.is_main_process:
        logger.info(f"Dataset loaded. Creating model: {cfg.model._target_}")

    model: UNetModel3D = instantiate(cfg.model)
    scheduler = instantiate(cfg.scheduler)

    if accelerator.is_main_process:
        logger.info("Creating trainer...")

    trainer: UNetTrainer = instantiate(
        cfg.trainer,
        train_set,
        model=model,
        accelerator=accelerator,
        scheduler=scheduler,
    )
    trainer.val_loader = val_set

    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("STARTING MULTI-EXPERIMENT TRAINING")
        print(f"Scenarios : {train_set.dataset.scenario_names}")
        print(f"Batch size: {data_cfg.batch_size}  ({data_cfg.batch_size // len(train_set.dataset.scenario_names)} per scenario)")
        print("=" * 70 + "\n")

    trainer.train()


if __name__ == "__main__":
    main()