#!/usr/bin/env python3

import os
import sys

import numpy as np
import torch
import xarray as xr

from accelerate import Accelerator
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Import from the repository's data package
from data.multi_experiment_dataset import build_multi_experiment_loader
from data.climate_dataset import set_minmax_override


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT = "runs/run_mseyb_BCprect_509.pt"

CONFIG_DIR = "configs"
MODEL_CONFIG = "config_aero"

# Scenario to use from the data YAML
SCENARIO = "ssp370"

# Which conditioning window to use
CONDITIONING_INDEX = 0

# Number of stochastic realizations for the same condition
N_SAMPLES = 10

# Number of reverse diffusion steps
SAMPLE_STEPS = 100

# Output file
OUTPUT = "generated_samples.nc"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD MAIN HYDRA CONFIG
# ============================================================

with initialize_config_dir(
    version_base=None,
    config_dir=os.path.abspath(CONFIG_DIR),
):
    cfg = compose(
        config_name=MODEL_CONFIG
    )

print("Loaded model config:", MODEL_CONFIG)


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

print("Loaded data config:", data_config_file)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint:")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",weights_only=False,
)

print("  ", CHECKPOINT)
print("Checkpoint keys:")

for key in checkpoint.keys():
    print("   ", key)


# ============================================================
# RESTORE CONDITIONING NORMALIZATION
# ============================================================

#
# IMPORTANT:
#
# This must happen BEFORE the dataset is constructed.
#
# The dataset will then use exactly the same CO2/SUL/BC
# normalization ranges that were used during training.
#

if "COND_NORM" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain COND_NORM. "
        "Cannot guarantee that conditioning is normalized "
        "the same way as during training."
    )

print("\nRestoring COND_NORM...")

set_minmax_override(
    checkpoint["COND_NORM"]
)

print("COND_NORM restored.")


# ============================================================
# ACCELERATE
# ============================================================

accelerator = Accelerator()


# ============================================================
# CREATE MODEL FROM CONFIG
# ============================================================

print("\nCreating model from config...")

model = instantiate(
    cfg.model
)

model = model.to(device)


# ============================================================
# LOAD EMA WEIGHTS
# ============================================================

print("Loading EMA weights...")

missing, unexpected = model.load_state_dict(
    checkpoint["EMA"],
    strict=False,
)

if missing:

    print("\nMissing model keys:")

    for key in missing:
        print("   ", key)

if unexpected:

    print("\nUnexpected model keys:")

    for key in unexpected:
        print("   ", key)

if missing or unexpected:

    raise RuntimeError(
        "Model checkpoint does not exactly match "
        "the model created from config_aero.yaml."
    )

model.eval()

print("EMA model loaded successfully.")


# ============================================================
# CREATE SCHEDULER FROM CONFIG
# ============================================================

print("\nCreating scheduler from config...")

scheduler = instantiate(
    cfg.scheduler
)

scheduler = scheduler.to(device)

print("Scheduler created.")


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

print("Dataset created.")


# ============================================================
# RESTORE PCA STATE
# ============================================================

#
# PCA is stored separately for the different scenarios.
#

if "PCA" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain PCA state."
    )

print("\nRestoring PCA state...")

loader.dataset.set_pca_state(
    checkpoint["PCA"]
)

print("PCA restored.")


# ============================================================
# FIND REQUESTED SCENARIO
# ============================================================

print(
    "\nAvailable scenarios:",
    loader.dataset.scenario_names,
)

if SCENARIO not in loader.dataset.scenario_names:

    raise ValueError(
        f"Scenario '{SCENARIO}' not found. "
        f"Available scenarios: "
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
    "\nLoading conditioning window..."
)

x, cond, scenario_id = (
    scenario_dataset[
        CONDITIONING_INDEX
    ]
)

#
# Dataset returns:
#
#     cond = [C, T, lat, lon]
#
# Add batch dimension:
#
#     [1, C, T, lat, lon]
#

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
# CONTINUOUS DIFFUSION SAMPLING
# ============================================================

@torch.inference_mode()
def generate_sample(
    cond,
    model,
    scheduler,
    sample_steps,
):
    """
    Generate one stochastic realization.

    This follows the continuous-time sampling formulation
    used by the production evaluator:

        t = 1 -> 0

        model input = scheduler.log_snr(t)

        v prediction
            ->
        x0 prediction
            ->
        q posterior
            ->
        next sample
    """

    batch_size = cond.shape[0]

    # --------------------------------------------------------
    # Output channels
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
        cond.shape[2],
        cond.shape[3],
        cond.shape[4],
    )

    # --------------------------------------------------------
    # Start from Gaussian noise
    # --------------------------------------------------------

    sample = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Continuous diffusion times
    #
    # 1.0 -> 0.0
    # --------------------------------------------------------

    steps = torch.linspace(
        1.0,
        0.0,
        sample_steps + 1,
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Reverse diffusion
    # --------------------------------------------------------

    for i in range(sample_steps):

        t = steps[i].expand(
            batch_size
        )

        t_next = steps[i + 1].expand(
            batch_size
        )

        # ----------------------------------------------------
        # Model predicts v
        #
        # IMPORTANT:
        #
        # The UNet receives log-SNR, not the integer
        # diffusion step.
        # ----------------------------------------------------

        model_output = model(
            sample,
            scheduler.log_snr(t),
            cond_map=cond,
        )

        # ----------------------------------------------------
        # Convert v prediction to x0
        # ----------------------------------------------------

        x_start = (
            scheduler.predict_start_from_v(
                sample,
                t,
                model_output,
            )
        )

        # ----------------------------------------------------
        # Posterior
        # ----------------------------------------------------

        mean, variance, _ = (
            scheduler.q_posterior(
                x_start,
                sample,
                t,
                t_next=t_next,
            )
        )

        # ----------------------------------------------------
        # Final step is deterministic
        # ----------------------------------------------------

        if i == sample_steps - 1:

            sample = mean

        else:

            noise = torch.randn_like(
                sample
            )

            sample = (
                mean
                + torch.sqrt(variance)
                * noise
            )

        print(
            f"\rDiffusion step "
            f"{i + 1}/{sample_steps}",
            end="",
            flush=True,
        )

    print()

    return sample


# ============================================================
# GENERATE ENSEMBLE
# ============================================================

print(
    f"\nGenerating {N_SAMPLES} "
    f"realizations for {SCENARIO}..."
)

generated = []

for i in range(N_SAMPLES):

    print(
        f"\nSample {i + 1}/{N_SAMPLES}"
    )

    sample = generate_sample(
        cond,
        model,
        scheduler,
        SAMPLE_STEPS,
    )

    generated.append(
        sample.cpu()
    )


# ============================================================
# COMBINE
# ============================================================

#
# Each sample:
#
#     [1, C, T, lat, lon]
#
# Combined:
#
#     [N_SAMPLES, C, T, lat, lon]
#

generated = torch.cat(
    generated,
    dim=0,
)

generated = generated.numpy()

print(
    "\nGenerated model-space shape:",
    generated.shape
)


# ============================================================
# DENORMALIZE TARGETS
# ============================================================

#
# The diffusion model output is still in normalized model
# space.
#
# Convert it back to physical units before writing NetCDF.
#

target_vars = OmegaConf.to_container(
    data_cfg.target_vars,
    resolve=True,
)

print(
    "Target variables:",
    target_vars
)


if len(target_vars) != 2:

    raise RuntimeError(
        "This generation script expects exactly "
        "two target variables: TREFHT and PRECT."
    )


# ------------------------------------------------------------
# TREFHT
# ------------------------------------------------------------

trefht_index = target_vars.index(
    "TREFHT"
)

trefht = (
    generated[:, trefht_index]
    * 21.0
    + 4.5
)


# ------------------------------------------------------------
# PRECT
# ------------------------------------------------------------

prect_index = target_vars.index(
    "PRECT"
)

prect = np.expm1(
    generated[:, prect_index]
    * 0.5703
    + 1.0727
)


# ============================================================
# GET COORDINATES
# ============================================================

dataset = scenario_dataset


# ------------------------------------------------------------
# Latitude
# ------------------------------------------------------------

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
        "Could not find latitude coordinate "
        "in ClimateDataset."
    )


# ------------------------------------------------------------
# Longitude
# ------------------------------------------------------------

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
        "Could not find longitude coordinate "
        "in ClimateDataset."
    )


# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

if hasattr(dataset, "time"):

    time = np.asarray(
        dataset.time
    )

else:

    #
    # Fallback.
    #
    # This is only used if ClimateDataset does not expose
    # the original time coordinate.
    #

    time = np.arange(
        generated.shape[2]
    )


time = time[
    :generated.shape[2]
]


# ============================================================
# CREATE NETCDF
# ============================================================

output_ds = xr.Dataset(

    data_vars={

        "TREFHT": (
            (
                "sample",
                "time",
                "lat",
                "lon",
            ),
            trefht,
        ),

        "PRECT": (
            (
                "sample",
                "time",
                "lat",
                "lon",
            ),
            prect,
        ),
    },

    coords={

        "sample": np.arange(
            N_SAMPLES
        ),

        "time": time,

        "lat": lat,

        "lon": lon,
    },
)


# ============================================================
# ADD VARIABLE ATTRIBUTES
# ============================================================

output_ds["TREFHT"].attrs = {
    "long_name": "Near-surface air temperature",
    "units": "degC",
}

output_ds["PRECT"].attrs = {
    "long_name": "Precipitation",
    "units": "mm day-1",
}


# ============================================================
# ADD GLOBAL ATTRIBUTES
# ============================================================

output_ds.attrs = {

    "description":
        "Diffusion-model generated climate realizations",

    "scenario":
        SCENARIO,

    "checkpoint":
        CHECKPOINT,

    "sample_steps":
        SAMPLE_STEPS,

    "n_samples":
        N_SAMPLES,

    "conditioning_index":
        CONDITIONING_INDEX,
}


# ============================================================
# SAVE NETCDF
# ============================================================

print(
    "\nWriting NetCDF:",
    OUTPUT
)

output_ds.to_netcdf(
    OUTPUT
)


# ============================================================
# FINISHED
# ============================================================

print("\nDone!")

print(
    "\nOutput:"
)

print(
    output_ds
)