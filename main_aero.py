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
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
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

    if accelerator.is_main_process:
        logger.info(f"Rank {accelerator.process_index}/{accelerator.num_processes}")
        logger.info("Building multi-experiment dataset...")

    # ── Build multi-experiment train dataset ────────────────────────────────
    # Reads from configs/conf_data.yaml (included via config_aero.yaml defaults)
    train_set: MultiExperimentDataset = build_multi_experiment_loader(
        experiment_configs=cfg.data.experiment_configs,
        accelerator=accelerator,
        batch_size=cfg.data.batch_size,
        mix_scenarios=cfg.data.get("mix_scenarios", True),
        steps_per_realization=cfg.data.get("steps_per_realization", None),
        # shared ClimateDataset kwargs
        seq_len=cfg.data.seq_len,
        target_vars=cfg.data.target_vars,
        cond_vars=cfg.data.cond_vars,
        n_components_target=cfg.data.get("n_components_target", None),
        n_components_cond=cfg.data.get("n_components_cond", None),
    )

    if accelerator.is_main_process:
        logger.info(f"Dataset loaded. Creating model: {cfg.model._target_}")

    model: UNetModel3D = instantiate(cfg.model)
    scheduler: DDPMScheduler = instantiate(cfg.scheduler)

    if accelerator.is_main_process:
        logger.info("Creating trainer...")

    trainer: UNetTrainer = instantiate(
        cfg.trainer,
        train_set,          # MultiExperimentDataLoader (has .dataset for metadata)
        model=model,
        accelerator=accelerator,
        scheduler=scheduler,
    )

    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("STARTING MULTI-EXPERIMENT TRAINING")
        scenarios = train_set.dataset.scenario_names
        print(f"Scenarios : {scenarios}")
        print(f"Batch size: {cfg.data.batch_size}  ({cfg.data.batch_size // len(scenarios)} per scenario)")
        print("=" * 70 + "\n")

    trainer.train()


if __name__ == "__main__":
    main()