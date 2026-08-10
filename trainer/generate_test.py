#!/usr/bin/env python3

import os

import numpy as np
import torch
import xarray as xr

from accelerate import Accelerator

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from omegaconf import OmegaConf

from multi_experiment_dataset import build_multi_experiment_loader


# ============================================================
# SETTINGS
# ============================================================

# Trained .pt checkpoint
CHECKPOINT = "runs/un_mseyb_BCprect_509.pt"

# Directory containing config_aero.yaml and data YAML
CONFIG_DIR = "configs"

# Main training configuration
MODEL_CONFIG = "config_aero"

# Which scenario to use
SCENARIO = "ssp370"

# Which conditioning sample/time window to use
CONDITIONING_INDEX = 0

# Number of stochastic realizations
N_SAMPLES = 10

# Number of reverse diffusion steps
SAMPLE_STEPS = 100

# Output NetCDF
OUTPUT = "generated_samples.nc"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD MAIN CONFIG
# ============================================================

with initialize_config_dir(
    version_base=None,
    config_dir=os.path.abspath(CONFIG_DIR),
):

    cfg = compose(
        config_name=MODEL_CONFIG
    )


# ============================================================
# LOAD DATA CONFIG
# ============================================================

data_config_file = os.path.join(
    CONFIG_DIR,
    cfg.data_config,
)

data_cfg = OmegaConf.load(
    data_config_file
)

print(
    "Data config:",
    data_config_file
)


# ============================================================
# ACCELERATE
# ============================================================

accelerator = Accelerator()


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating model...")

model = instantiate(
    cfg.model
)

model = model.to(device)


# ============================================================
# CREATE SCHEDULER
# ============================================================

print("Creating scheduler...")

scheduler = instantiate(
    cfg.scheduler
)

scheduler = scheduler.to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(
    "\nLoading checkpoint:",
    CHECKPOINT
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

print(
    "Checkpoint contents:"
)

for key in checkpoint.keys():
    print("   ", key)


# ============================================================
# LOAD EMA MODEL
# ============================================================

print("\nLoading EMA model...")

model.load_state_dict(
    checkpoint["EMA"],
    strict=True,
)

model.eval()

print("EMA model loaded.")


# ============================================================
# BUILD DATASET
# ============================================================

print("\nBuilding dataset...")

experiment_configs = OmegaConf.to_container(
    data_cfg.experiment_configs,
    resolve=True,
)

loader = build_multi_experiment_loader(

    experiment_configs=experiment_configs,

    accelerator=accelerator,

    batch_size=1,

    # We want to select the scenario ourselves.
    mix_scenarios=False,

    seq_len=data_cfg.seq_len,

    target_vars=OmegaConf.to_container(
        data_cfg.target_vars,
        resolve=True,
    ),

    cond_vars=OmegaConf.to_container(
        data_cfg.cond_vars,
        resolve=True,
    ),

    n_components_target=data_cfg.get(
        "n_components_target",
        None,
    ),

    n_components_cond=data_cfg.get(
        "n_components_cond",
        None,
    ),

    cond_smooth_sigma=data_cfg.get(
        "cond_smooth_sigma",
        None,
    ),

    cond_smooth_method=data_cfg.get(
        "cond_smooth_method",
        "gaussian",
    ),

    shard_across_ranks=False,
)


# ============================================================
# RESTORE PCA
# ============================================================

if "PCA" in checkpoint:

    print("\nRestoring PCA state...")

    loader.dataset.set_pca_state(
        checkpoint["PCA"]
    )

    print("PCA restored.")

else:

    print(
        "\nWARNING:"
        " checkpoint does not contain PCA state."
    )


# ============================================================
# FIND SCENARIO
# ============================================================

print(
    "\nAvailable scenarios:",
    loader.dataset.scenario_names,
)

if SCENARIO not in loader.dataset.scenario_names:

    raise ValueError(
        f"Scenario '{SCENARIO}' not found. "
        f"Available: "
        f"{loader.dataset.scenario_names}"
    )


scenario_index = (
    loader.dataset.scenario_names.index(
        SCENARIO
    )
)

scenario_dataset = (
    loader.dataset.datasets[
        scenario_index
    ]
)

print(
    "Using scenario:",
    SCENARIO
)


# ============================================================
# GET CONDITIONING MAP
# ============================================================

print(
    "\nGetting conditioning data..."
)

x, cond, scenario_id = (
    scenario_dataset[
        CONDITIONING_INDEX
    ]
)

# Add batch dimension
#
# cond:
#     [C, T, lat, lon]
#
# becomes:
#     [1, C, T, lat, lon]

cond = cond.unsqueeze(0)

cond = cond.to(
    device=device,
    dtype=torch.float32,
)

print(
    "Conditioning shape:",
    tuple(cond.shape)
)


# ============================================================
# GENERATION FUNCTION
# ============================================================

@torch.no_grad()
def generate_samples(cond_map):

    batch_size = cond_map.shape[0]

    # --------------------------------------------------------
    # Number of target variables
    # --------------------------------------------------------

    n_channels = len(
        data_cfg.target_vars
    )

    # --------------------------------------------------------
    # Output shape
    #
    # [batch, channels, time, lat, lon]
    # --------------------------------------------------------

    shape = (
        batch_size,
        n_channels,
        cond_map.shape[2],
        cond_map.shape[3],
        cond_map.shape[4],
    )

    # --------------------------------------------------------
    # Start from random Gaussian noise
    # --------------------------------------------------------

    sample = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Configure scheduler
    # --------------------------------------------------------

    scheduler.set_timesteps(
        SAMPLE_STEPS
    )

    # --------------------------------------------------------
    # Reverse diffusion
    # --------------------------------------------------------

    for i, timestep in enumerate(
        scheduler.timesteps
    ):

        # Model expects a timestep for
        # every item in the batch.

        t = torch.full(
            (batch_size,),
            int(timestep),
            device=device,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Predict v
        # ----------------------------------------------------

        model_output = model(
            sample,
            t,
            cond_map=cond_map,
        )

        # ----------------------------------------------------
        # Reverse diffusion
        # ----------------------------------------------------

        result = scheduler.step(
            model_output,
            int(timestep),
            sample,
        )

        sample = result.prev_sample

        print(
            f"\rDiffusion step "
            f"{i + 1}/{SAMPLE_STEPS}",
            end="",
            flush=True,
        )

    print()

    return sample


# ============================================================
# GENERATE ENSEMBLE
# ============================================================

generated = []

print(
    f"\nGenerating {N_SAMPLES} "
    f"realizations for {SCENARIO}..."
)

for i in range(N_SAMPLES):

    print(
        f"\nSample {i + 1}/{N_SAMPLES}"
    )

    sample = generate_samples(
        cond
    )

    generated.append(
        sample.cpu()
    )


# ============================================================
# COMBINE SAMPLES
# ============================================================

#
# Each sample:
#
#     [1, C, T, lat, lon]
#
# After concatenation:
#
#     [N_SAMPLES, C, T, lat, lon]
#

generated = torch.cat(
    generated,
    dim=0,
)

generated = generated.numpy()

print(
    "\nGenerated array shape:",
    generated.shape
)


# ============================================================
# GET COORDINATES
# ============================================================

#
# Get coordinates from the underlying ClimateDataset.
#

dataset = scenario_dataset


# Try the standard coordinate names first.

if hasattr(dataset, "lat"):
    lat = np.asarray(
        dataset.lat
    )
elif hasattr(dataset, "latitude"):
    lat = np.asarray(
        dataset.latitude
    )
else:
    raise AttributeError(
        "Could not find latitude coordinate."
    )


if hasattr(dataset, "lon"):
    lon = np.asarray(
        dataset.lon
    )
elif hasattr(dataset, "longitude"):
    lon = np.asarray(
        dataset.longitude
    )
else:
    raise AttributeError(
        "Could not find longitude coordinate."
    )


# ------------------------------------------------------------
# Time coordinate
# ------------------------------------------------------------

if hasattr(dataset, "time"):

    time = np.asarray(
        dataset.time
    )

else:

    # If ClimateDataset does not expose time directly,
    # create an integer time coordinate.

    time = np.arange(
        generated.shape[2]
    )


# ============================================================
# CREATE NETCDF
# ============================================================

target_vars = OmegaConf.to_container(
    data_cfg.target_vars,
    resolve=True,
)

print(
    "\nTarget variables:",
    target_vars
)


data_vars = {}

for i, variable in enumerate(
    target_vars
):

    data_vars[variable] = (
        (
            "sample",
            "time",
            "lat",
            "lon",
        ),
        generated[:, i, :, :, :],
    )


output_ds = xr.Dataset(

    data_vars=data_vars,

    coords={
        "sample": np.arange(
            generated.shape[0]
        ),

        "time": time[
            :generated.shape[2]
        ],

        "lat": lat,

        "lon": lon,
    },
)


# ============================================================
# METADATA
# ============================================================

output_ds.attrs[
    "scenario"
] = SCENARIO

output_ds.attrs[
    "source_checkpoint"
] = CHECKPOINT

output_ds.attrs[
    "n_diffusion_steps"
] = SAMPLE_STEPS

output_ds.attrs[
    "description"
] = (
    "Diffusion-model generated "
    "climate realizations"
)


# ============================================================
# SAVE
# ============================================================

print(
    "\nWriting:",
    OUTPUT
)

output_ds.to_netcdf(
    OUTPUT
)

print(
    "\nDone!"
)

print(
    output_ds
)