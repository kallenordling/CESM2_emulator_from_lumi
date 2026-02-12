# Standard library imports
import os
from typing import Union
from collections import OrderedDict
from hydra.utils import instantiate
# Third party imports
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import hydra
from omegaconf.omegaconf import DictConfig
from hydra.utils import instantiate
from accelerate import Accelerator
from diffusers import DDPMScheduler
import xarray as xr
from tqdm import tqdm
import pandas as pd
from custom_diffusers.continuous_ddpm import ContinuousDDPM
# Local imports
from data.climate_dataset import ClimateDataset
from utils.gen_utils import generate_samples, generate_samples2
from omegaconf import OmegaConf
from models.video_net import UNetModel3D
from ema_pytorch import EMA

Checkpoint = dict[str, Union[int, OrderedDict]]

# Assumes that gen is conditioned on val, and the first realization
# is always reserved for the test set
realization_dict = {"gen": "r10i1181p1f1", "val": "r10i1181p1f1", "test": "r10i1181p1f1"}


def get_starting_index(directory: str) -> int:
    """Goes through a directory of files named "member_i.nc" and returns the next available index."""
    files = os.listdir(directory)
    indices = [
        int(file.split("_")[1].split(".")[0])
        for file in files
        if file.startswith("member")
    ]
    return max(indices) + 1 if indices else 0


def create_batches(
        xr_ds: xr.Dataset,
        dataset: ClimateDataset,
) -> list[xr.Dataset]:
    """Splits the dataset up into batches of size batch_size."""
    seq_len = dataset.seq_len

    data = []
    for i in range(0, len(xr_ds.year), seq_len):
        batch = xr_ds.isel(year=slice(i, i + seq_len))
        tensor_data = dataset.convert_xarray_to_tensor(batch)
        data.append((tensor_data, dict(batch.coords)))

    return data


def custom_collate_fn(
        batches: list[tuple[Tensor, xr.DataArray]],
) -> tuple[Tensor, list[xr.DataArray]]:
    """Collate function for the dataloader."""
    tensor_batch = []
    coords = []
    for batch in batches:
        tensor_batch.append(batch[0])
        coords.append(batch[1])

    return torch.stack(tensor_batch), coords


@hydra.main(version_base=None, config_path="./configs", config_name="generate_aero_ssp370")
def main(config: DictConfig) -> None:
    # Verify that the save folder exists
    assert os.path.isdir(config.save_dir), "Save directory does not exist"
    assert config.gen_mode in ["gen", "val", "test"], "Invalid gen mode"

    if config.gen_mode == "gen":
        assert config.load_path, "Must specify a load path"
        print(config.load_path)
        assert os.path.isfile(config.load_path), "Invalid load path"

    assert (
            config.samples_per == 1 or config.gen_mode == "gen"
    ), "Number of samples must be 1 for val and test"

    # Read guidance_scale from config, default to 1.0 (no guidance)
    guidance_scale = getattr(config, "guidance_scale", 1.0)
    print(f"[GENERATE] guidance_scale = {guidance_scale}")

    # Initialize all necessary objects
    accelerator = Accelerator(**config.accelerator)

    dataset: ClimateDataset = instantiate(
        config.dataset,
        data_dir=config.data_dir,
        realizations=[realization_dict[config.gen_mode]],
        target_vars=config.variables, cond_vars=["CO2", 'SO2'], cond_file=config.cond_file
    )
    scheduler: ContinuousDDPM = instantiate(config.scheduler)
    scheduler.set_timesteps(config.sample_steps)

    conf = OmegaConf.load('./configs/config_aero.yaml')
    model_conf: UNetModel3D = instantiate(conf.model)

    if config.gen_mode == "gen":
        # Load the model from the checkpoint
        chkpt: Checkpoint = torch.load(config.load_path, map_location="cpu", weights_only=False)

        ema_model_sd = chkpt["EMA"]

        model = EMA(
            model_conf,
            beta=0.9999,
            update_after_step=100,
            update_every=10,
        ).to(accelerator.device)
        model.ema_model.load_state_dict(ema_model_sd)
        model.eval()

    else:
        model = None

    # Grab the Xarray dataset from the dataset object
    xr_ds = dataset.dataset_cond.load()
    print(xr_ds)
    xr_ds = xr_ds.sel(year=slice(str(config.start_year), str(config.end_year)))

    batches = create_batches(xr_ds, dataset)

    dataloader = DataLoader(
        batches, batch_size=config.batch_size, collate_fn=custom_collate_fn
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    for i in tqdm(range(config.samples_per)):
        gen_samples = []

        for tensor_batch, coords in tqdm(
                dataloader, disable=not accelerator.is_main_process
        ):
            tensor_batch = tensor_batch.to(accelerator.device)

            if model is not None:
                gen_months, sal_co2, sal_sul = generate_samples2(
                    tensor_batch, tensor_batch,
                    scheduler=scheduler,
                    sample_steps=config.sample_steps,
                    model=model,
                    disable=True,
                    guidance_scale=guidance_scale,
                )
            else:
                gen_months = tensor_batch

            for i in range(len(gen_months)):
                sample = gen_months[i]  # shape: (1, T, H, W)

                print(f"Sample shape before conversion: {sample.shape}")
                print(f"Expected variables: {dataset.vars}")
                print(f"Number of variables: {len(dataset.vars)}")

                original_vars = dataset.vars
                dataset.vars = dataset.vars[:sample.shape[0]]

                gen_samples.append(
                    dataset.convert_tensor_to_xarray(sample, coords=coords[i])
                )

                dataset.vars = original_vars

        gen_samples = accelerator.gather_for_metrics(gen_samples)
        gen_samples = xr.concat(gen_samples, "year").sortby("year")

        if accelerator.is_main_process:
            save_name = f"{config.gen_mode}_{config.save_name + '_' if config.save_name is not None else ''}{'_'.join(config.variables)}_{config.start_year}-{config.end_year}.nc"
            save_path = os.path.join(
                config.data_dir, save_name
            )
            if config.gen_mode == "gen" and config.samples_per > 1:
                save_dir = save_path.strip(".nc")
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)

                mem_index = get_starting_index(save_dir)
                save_path = os.path.join(save_dir, f"member_{mem_index}.nc")

            else:
                if os.path.isfile(save_path):
                    os.remove(save_path)

            print('save file', save_path)
            gen_samples.to_netcdf(save_path)
            os.chmod(save_path, 0o770)


if __name__ == "__main__":
    main()