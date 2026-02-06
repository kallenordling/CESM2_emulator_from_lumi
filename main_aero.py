from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import DDPMScheduler

from data.climate_dataset import ClimateDataset
from trainer.unetTrainer import UNetTrainer
from models.video_net import UNetModel3D

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed')
warnings.filterwarnings('ignore', category=FutureWarning)
@hydra.main(version_base=None, config_path="configs", config_name="config_aero.yaml")
def main(cfg: DictConfig) -> None:
    """
    Minimal main function that avoids barriers until absolutely necessary.
    Use this version if you're experiencing NCCL connection issues.
    """

    # Simple accelerator configuration
    # Let SLURM/environment handle most of the setup
    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.mixed_precision,
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        split_batches=cfg.accelerator.get('split_batches', False),
    )

    # Set seed
    set_seed(cfg.seed, device_specific=False)

    # Logger
    logger = get_logger(__name__, log_level="INFO")

    # Only print from main process
    if accelerator.is_main_process:
        logger.info(f"Rank {accelerator.process_index}/{accelerator.num_processes}")
        logger.info(f"Loading dataset: {cfg.data.train._target_}")

    # Load dataset - NO barriers, each rank loads independently
    train_cfg = dict(cfg.data.train)
    train_set: ClimateDataset = instantiate(train_cfg, _recursive_=False)

    if accelerator.is_main_process:
        logger.info(f"Dataset loaded. Creating model: {cfg.model._target_}")

    # Create model
    model: UNetModel3D = instantiate(cfg.model)

    if accelerator.is_main_process:
        logger.info(f"Model created. Creating scheduler: {cfg.scheduler._target_}")

    # Create scheduler
    scheduler: DDPMScheduler = instantiate(cfg.scheduler)

    if accelerator.is_main_process:
        logger.info(f"Creating trainer: {cfg.trainer._target_}")

    # Create trainer
    trainer: UNetTrainer = instantiate(
        cfg.trainer,
        train_set,
        model=model,
        accelerator=accelerator,
        scheduler=scheduler,
    )

    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70 + "\n")

    # First barrier happens inside trainer.train() when it calls prepare()
    trainer.train()


if __name__ == "__main__":
    main()