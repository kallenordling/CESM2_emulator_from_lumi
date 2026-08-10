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

# Scalar coefficients for the EBM auxiliary loss. unetTrainer.py:386-391
# creates them as Parameters and register_parameter()s them ONTO the model, so
# any checkpoint trained with the EBM term carries them — but they take no part
# in the diffusion forward pass, so they are expected extras at inference and
# safe to drop. eval_aero.py ignores them the same way (strict=False).
TRAINER_ONLY_KEYS = {
    "ebm_alpha_ghg",
    "ebm_alpha_aero",
    "ebm_lambda",
}

unexpected_real = [
    key for key in unexpected
    if key not in TRAINER_ONLY_KEYS
]

if missing:

    print("\nMissing model keys:")

    for key in missing:
        print("   ", key)

if unexpected:

    print("\nUnexpected model keys:")

    for key in unexpected:

        note = (
            "  (trainer-only, ignored)"
            if key in TRAINER_ONLY_KEYS
            else ""
        )

        print("   ", key + note)

# MISSING keys are always fatal: the model would run with zero-initialised
# weights and silently produce garbage. Extra keys are only fatal when they are
# not the known trainer-side ones.
if missing or unexpected_real:

    raise RuntimeError(
        "Model checkpoint does not match the model created from "
        f"{MODEL_CONFIG}.yaml — "
        f"{len(missing)} missing, "
        f"{len(unexpected_real)} unexpected "
        "(excluding known trainer-only keys). "
        "Check that the model config's in/out/cond channel counts match "
        "the arm this checkpoint came from."
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

# scenario_dataset is the inner ClimateDataset, whose __getitem__ returns
# (x, cond) — 2 values (data/climate_dataset.py:1078). Only the OUTER
# MultiExperimentDataset appends scenario_id, by wrapping this call
# (data/multi_experiment_dataset.py:171-175). We indexed into .datasets[...]
# to pick the scenario ourselves, so we unpack 2 and rebuild scenario_id from
# the index we already resolved.
if CONDITIONING_INDEX >= len(scenario_dataset):

    raise IndexError(
        f"CONDITIONING_INDEX={CONDITIONING_INDEX} is out of range for "
        f"scenario '{SCENARIO}', which has {len(scenario_dataset)} "
        f"conditioning windows in the currently loaded realization."
    )

x, cond = scenario_dataset[
    CONDITIONING_INDEX
]

scenario_id = torch.tensor(
    scenario_index,
    dtype=torch.long,
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


#
# ClimateDataset does NOT expose .lat / .lon / .time. What it has is:
#
#     .xr_data       the loaded xarray Dataset (lat, lon and the time dim)
#                    — set at data/climate_dataset.py:641
#     .lats          latitude only, plural (:637)
#     ._time_values  integer years (:657)
#     .time_dim      name of the time dimension
#
# Prefer .xr_data, which carries all three on the same object.
#

lat = lon = None

xr_data = getattr(dataset, "xr_data", None)

if xr_data is not None:

    for name in ("lat", "latitude"):
        if name in xr_data.coords:
            lat = np.asarray(xr_data[name].values)
            break

    for name in ("lon", "longitude"):
        if name in xr_data.coords:
            lon = np.asarray(xr_data[name].values)
            break


# Fallback for latitude: the dataset keeps it separately as .lats (plural).

if lat is None and hasattr(dataset, "lats"):
    lat = np.asarray(dataset.lats)


if lat is None or lon is None:

    raise AttributeError(
        "Could not resolve lat/lon from the ClimateDataset. "
        f"lat={'ok' if lat is not None else 'MISSING'}, "
        f"lon={'ok' if lon is not None else 'MISSING'}. "
        "Expected .xr_data (with lat/lon coords) or .lats; "
        f"available attributes: "
        f"{[a for a in vars(dataset) if not a.startswith('__')][:20]}"
    )


# ------------------------------------------------------------
# Time
# ------------------------------------------------------------
#
# CONDITIONING_INDEX selects a window of length seq_len starting at that
# index, so the generated time axis is the matching slice of the dataset's
# own year values — not a slice from 0.
#

time_values = getattr(dataset, "_time_values", None)

if time_values is not None:

    time = np.asarray(time_values)[
        CONDITIONING_INDEX:
        CONDITIONING_INDEX + generated.shape[2]
    ]

elif xr_data is not None and getattr(dataset, "time_dim", None) in xr_data.coords:

    time = np.asarray(
        xr_data[dataset.time_dim].values
    )[
        CONDITIONING_INDEX:
        CONDITIONING_INDEX + generated.shape[2]
    ]

else:

    # Last resort: positional index, so the file is still writable.
    print(
        "WARNING: could not resolve real time values — "
        "writing a positional time index instead."
    )

    time = np.arange(
        generated.shape[2]
    )


time = np.asarray(time)[:generated.shape[2]]

print("Coordinates:")
print("   lat ", lat.shape, f"{lat[0]:.2f} .. {lat[-1]:.2f}")
print("   lon ", lon.shape, f"{lon[0]:.2f} .. {lon[-1]:.2f}")
print("   time", time.shape, list(time[:5]))


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