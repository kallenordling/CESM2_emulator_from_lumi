from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import DDPMScheduler
import torch
import os

from data.climate_dataset import ClimateDataset
from trainer.unetTrainer import UNetTrainer
from models.video_net import UNetModel3D


@hydra.main(version_base=None, config_path="configs", config_name="config_aero.yaml")
def main(cfg: DictConfig) -> None:
    # Set environment variable to increase NCCL timeout (default is 10 min)
    os.environ.setdefault('NCCL_TIMEOUT', '3600')  # 1 hour
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')

    # Create accelerator object and set RNG seed
    accelerator = Accelerator(**cfg.accelerator)
    set_seed(cfg.seed, device_specific=False)

    # Logger works with distributed processes
    logger = get_logger(__name__, log_level="INFO")

    # Init logger
    if accelerator.is_main_process:
        logger.info(f"Instantiating datasets <{cfg.data.train._target_}>")

    train_cfg = dict(cfg.data.train)

    # Load dataset on ALL ranks simultaneously (no main_process_first barrier)
    # This is safer for large-scale distributed training
    train_set: ClimateDataset = instantiate(
        train_cfg, _recursive_=False
    )

    # Explicit barrier after dataset loading
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info(f"Dataset loaded successfully on all ranks")
        logger.info(f"Instantiating model <{cfg.model._target_}>")

    model: UNetModel3D = instantiate(cfg.model)

    if accelerator.is_main_process:
        logger.info(str(model))

    if accelerator.is_main_process:
        logger.info(f"Instantiating scheduler <{cfg.scheduler._target_}>")

    scheduler: DDPMScheduler = instantiate(cfg.scheduler)

    if accelerator.is_main_process:
        logger.info(f"Instantiating Trainer <{cfg.trainer._target_}>")

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

    trainer.train()


if __name__ == "__main__":
    main()