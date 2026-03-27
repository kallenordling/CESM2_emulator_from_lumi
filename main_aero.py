import os
# Must be set before any GPU allocation (before torch import)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from accelerate import Accelerator
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

    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.mixed_precision,
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        split_batches=cfg.accelerator.get('split_batches', False),
    )

    set_seed(cfg.seed, device_specific=False)
    logger = get_logger(__name__, log_level="INFO")

    # ── Load data config directly from its own yaml file ───────────────────
    # The data config is flat (no `data:` wrapper) so we load it separately
    # rather than composing it into config_aero.yaml via defaults.
    data_cfg_path = os.path.join(
        hydra.utils.get_original_cwd(), "configs", cfg.data_config
    )
    data_cfg = OmegaConf.load(data_cfg_path)

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
        # shared ClimateDataset kwargs
        seq_len=data_cfg.seq_len,
        target_vars=OmegaConf.to_container(data_cfg.target_vars, resolve=True),
        cond_vars=OmegaConf.to_container(data_cfg.cond_vars, resolve=True),
        n_components_target=data_cfg.get("n_components_target", None),
        n_components_cond=data_cfg.get("n_components_cond", None),
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
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
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