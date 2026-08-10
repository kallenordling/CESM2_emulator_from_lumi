#!/usr/bin/env python3

import os

import numpy as np
import torch
import xarray as xr

from accelerate import Accelerator
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from data.multi_experiment_dataset import build_multi_experiment_loader
from data.climate_dataset import set_minmax_override


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT = "runs/run_mseyb_BCprect_509.pt"

CONFIG_DIR = "configs"
MODEL_CONFIG = "config_aero"

# Number of stochastic realizations PER YEAR
N_SAMPLES = 10

# Number of reverse diffusion steps
SAMPLE_STEPS = 100

# Output NetCDF
OUTPUT = "generated_samples_1850_2100.nc"


# ============================================================
# YEARS TO GENERATE
# ============================================================

#
# ALL_YEARS=True builds the loader with EvalClimateDataset instead of
# ClimateDataset, so every year present in the files is loaded.
#
# ClimateDataset (the TRAINING class) subsamples in load_data(): every 5th
# historical year and every other future year, ~76 of 251. That is deliberate —
# it is what every checkpoint was fitted on — but it leaves 5-year gaps when
# generating a continuous timeseries. EvalClimateDataset overrides only the year
# selection; normalisation, smoothing, PCA and tensor layout are identical.
#
ALL_YEARS = True

# Requested years. Anything the files do not actually contain is dropped with a
# printed note when USE_DATASET_YEARS is True (see the verification block).
SCENARIO_YEARS = {
    "hist": np.arange(1850, 2015),
    "ssp370": np.arange(2015, 2101),
}

USE_DATASET_YEARS = True


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

print(
    "Loaded model config:",
    MODEL_CONFIG
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
    "Loaded data config:",
    data_config_file
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint:")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

print("  ", CHECKPOINT)

print("Checkpoint keys:")

for key in checkpoint.keys():
    print("   ", key)


# ============================================================
# RESTORE CONDITIONING NORMALIZATION
# ============================================================

#
# This MUST happen before the dataset is constructed.
#

if "COND_NORM" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain COND_NORM. "
        "Cannot reproduce the conditioning normalization "
        "used during training."
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
# CREATE MODEL
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


# These are created by the trainer for the EBM auxiliary
# loss but are not used by the diffusion forward pass.

TRAINER_ONLY_KEYS = {
    "ebm_alpha_ghg",
    "ebm_alpha_aero",
    "ebm_lambda",
}


unexpected_real = [
    key
    for key in unexpected
    if key not in TRAINER_ONLY_KEYS
]


if missing:

    print("\nMissing model keys:")

    for key in missing:
        print("   ", key)


if unexpected:

    print("\nUnexpected model keys:")

    for key in unexpected:

        if key in TRAINER_ONLY_KEYS:

            print(
                "   ",
                key,
                "(trainer-only, ignored)"
            )

        else:

            print(
                "   ",
                key
            )


if missing or unexpected_real:

    raise RuntimeError(
        "Model checkpoint does not match "
        "the model created from config."
    )


model.eval()

print(
    "EMA model loaded successfully."
)


# ============================================================
# CREATE SCHEDULER
# ============================================================

print(
    "\nCreating scheduler from config..."
)

scheduler = instantiate(
    cfg.scheduler
)

scheduler = scheduler.to(device)

print(
    "Scheduler created."
)


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

    # EvalClimateDataset instead of the training ClimateDataset -> every year
    all_years=ALL_YEARS,
)

print(
    "Dataset created."
)


# ============================================================
# RESTORE PCA
# ============================================================

if "PCA" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain PCA state."
    )

print(
    "\nRestoring PCA state..."
)

loader.dataset.set_pca_state(
    checkpoint["PCA"]
)

print(
    "PCA restored."
)


# ============================================================
# CHECK SCENARIOS
# ============================================================

print(
    "\nAvailable scenarios:",
    loader.dataset.scenario_names,
)


for scenario in SCENARIO_YEARS:

    if scenario not in loader.dataset.scenario_names:

        raise ValueError(
            f"Required scenario '{scenario}' "
            f"not found. Available scenarios: "
            f"{loader.dataset.scenario_names}"
        )


# ============================================================
# GET DATASETS
# ============================================================

hist_index = (
    loader.dataset.scenario_names.index(
        "hist"
    )
)

ssp370_index = (
    loader.dataset.scenario_names.index(
        "ssp370"
    )
)

hist_dataset = (
    loader.dataset.datasets[
        hist_index
    ]
)

ssp370_dataset = (
    loader.dataset.datasets[
        ssp370_index
    ]
)


# ============================================================
# PRINT AVAILABLE YEARS
# ============================================================

print("\nDataset year ranges:")

hist_available_years = np.asarray(
    hist_dataset._time_values
)

ssp370_available_years = np.asarray(
    ssp370_dataset._time_values
)

print(
    "  hist:",
    hist_available_years.min(),
    "-",
    hist_available_years.max(),
)

print(
    "  ssp370:",
    ssp370_available_years.min(),
    "-",
    ssp370_available_years.max(),
)


# ============================================================
# VERIFY REQUESTED YEARS EXIST
# ============================================================

for scenario, years in SCENARIO_YEARS.items():

    if scenario == "hist":

        available = hist_available_years

    else:

        available = ssp370_available_years


    missing_years = [
        int(year)
        for year in years
        if year not in available
    ]


    if missing_years and USE_DATASET_YEARS:

        # Fall back to exactly what this scenario provides, and say so loudly —
        # silently generating a different set of years than requested would be
        # worse than either erroring or reporting it.
        kept = sorted(int(y) for y in available)

        print(
            f"\nNOTE: {scenario} — {len(missing_years)} of "
            f"{len(years)} requested years are not in the dataset "
            f"(it subsamples: hist every 5th year, future every other; "
            f"data/climate_dataset.py:585-586)."
        )
        print(
            f"      Generating the {len(kept)} years it does provide: "
            f"{kept[0]}..{kept[-1]}"
        )

        SCENARIO_YEARS[scenario] = np.array(kept)

    elif missing_years:

        raise RuntimeError(
            f"{scenario} is missing "
            f"{len(missing_years)} requested years. "
            f"First missing years: "
            f"{missing_years[:10]}. "
            f"The dataset subsamples (hist every 5th year, future every "
            f"other; data/climate_dataset.py:585-586) — set "
            f"USE_DATASET_YEARS = True to generate what it provides, or "
            f"build the cond tensor directly from the cond NetCDF as "
            f"eval_aero.build_cond_tensor does to cover every year."
        )


# ============================================================
# TARGET VARIABLES
# ============================================================

target_vars = OmegaConf.to_container(
    data_cfg.target_vars,
    resolve=True,
)

print(
    "\nTarget variables:",
    target_vars
)


if target_vars != ["TREFHT", "PRECT"]:

    raise RuntimeError(
        "This script expects target_vars to be "
        "['TREFHT', 'PRECT']."
    )


# ============================================================
# GENERATION FUNCTION
# ============================================================

@torch.inference_mode()
def generate_samples(
    cond,
    n_samples,
):
    """
    Generate n_samples stochastic realizations
    for one conditioning map.

    cond:
        [1, cond_channels, time, lat, lon]

    Returns:
        [n_samples, 2, time, lat, lon]
    """

    # --------------------------------------------------------
    # Repeat conditioning for the whole ensemble
    # --------------------------------------------------------

    cond_batch = cond.repeat(
        n_samples,
        1,
        1,
        1,
        1,
    )

    batch_size = n_samples

    # --------------------------------------------------------
    # Output shape
    # --------------------------------------------------------

    n_channels = len(
        data_cfg.target_vars
    )

    shape = (
        batch_size,
        n_channels,
        cond.shape[2],
        cond.shape[3],
        cond.shape[4],
    )

    # --------------------------------------------------------
    # Start from independent Gaussian noise
    # --------------------------------------------------------

    sample = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Continuous diffusion times
    #
    # 1 -> 0
    # --------------------------------------------------------

    steps = torch.linspace(
        1.0,
        0.0,
        SAMPLE_STEPS + 1,
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Reverse diffusion
    # --------------------------------------------------------

    for i in range(SAMPLE_STEPS):

        t = steps[i].expand(
            batch_size
        )

        t_next = steps[i + 1].expand(
            batch_size
        )

        # ----------------------------------------------------
        # Model predicts v
        # ----------------------------------------------------

        model_output = model(
            sample,
            scheduler.log_snr(t),
            cond_map=cond_batch,
        )

        # ----------------------------------------------------
        # v -> x0
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
        # Final step deterministic
        # ----------------------------------------------------

        if i == SAMPLE_STEPS - 1:

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
            f"{i + 1}/{SAMPLE_STEPS}",
            end="",
            flush=True,
        )

    print()

    return sample.cpu()


# ============================================================
# COORDINATES
# ============================================================

#
# Use the historical dataset for the common grid.
#

coordinate_dataset = hist_dataset

xr_data = getattr(
    coordinate_dataset,
    "xr_data",
    None,
)


# ------------------------------------------------------------
# Latitude
# ------------------------------------------------------------

lat = None

if xr_data is not None:

    for name in (
        "lat",
        "latitude",
    ):

        if name in xr_data.coords:

            lat = np.asarray(
                xr_data[name].values
            )

            break


if lat is None and hasattr(
    coordinate_dataset,
    "lats",
):

    lat = np.asarray(
        coordinate_dataset.lats
    )


# ------------------------------------------------------------
# Longitude
# ------------------------------------------------------------

lon = None

if xr_data is not None:

    for name in (
        "lon",
        "longitude",
    ):

        if name in xr_data.coords:

            lon = np.asarray(
                xr_data[name].values
            )

            break


if lat is None or lon is None:

    raise RuntimeError(
        "Could not find lat/lon coordinates."
    )


print(
    "\nGrid:",
    len(lat),
    "x",
    len(lon),
)


# ============================================================
# CREATE OUTPUT FILE
# ============================================================

#
# We create the complete output structure first.
#
# Dimensions:
#
#     year
#     sample
#     lat
#     lon
#
# Each variable:
#
#     [year, sample, lat, lon]
#
# This gives:
#
#     1850 ... 2100
#       10 samples/year
#
# ============================================================

all_years = np.arange(
    1850,
    2101,
)

n_years = len(
    all_years
)


# ------------------------------------------------------------
# Create empty arrays on disk via NetCDF4
# ------------------------------------------------------------

try:

    from netCDF4 import Dataset

except ImportError:

    raise ImportError(
        "This script requires netCDF4. "
        "Install it with: pip install netCDF4"
    )


print(
    "\nCreating output:",
    OUTPUT
)


nc = Dataset(
    OUTPUT,
    "w",
)


# ------------------------------------------------------------
# Dimensions
# ------------------------------------------------------------

nc.createDimension(
    "year",
    n_years,
)

nc.createDimension(
    "sample",
    N_SAMPLES,
)

nc.createDimension(
    "lat",
    len(lat),
)

nc.createDimension(
    "lon",
    len(lon),
)


# ------------------------------------------------------------
# Coordinates
# ------------------------------------------------------------

year_var = nc.createVariable(
    "year",
    "i4",
    ("year",),
)

sample_var = nc.createVariable(
    "sample",
    "i4",
    ("sample",),
)

lat_var = nc.createVariable(
    "lat",
    "f4",
    ("lat",),
)

lon_var = nc.createVariable(
    "lon",
    "f4",
    ("lon",),
)


year_var[:] = all_years
sample_var[:] = np.arange(
    N_SAMPLES
)

lat_var[:] = lat
lon_var[:] = lon


lat_var.units = "degrees_north"
lon_var.units = "degrees_east"


# ------------------------------------------------------------
# Target variables
# ------------------------------------------------------------

trefht_var = nc.createVariable(
    "TREFHT",
    "f4",
    ("year", "sample", "lat", "lon"),
    zlib=True,
    complevel=4,
)

prect_var = nc.createVariable(
    "PRECT",
    "f4",
    ("year", "sample", "lat", "lon"),
    zlib=True,
    complevel=4,
)


trefht_var.long_name = (
    "Near-surface air temperature"
)

trefht_var.units = "degC"


prect_var.long_name = (
    "Precipitation"
)

prect_var.units = "mm day-1"


# ------------------------------------------------------------
# Global attributes
# ------------------------------------------------------------

nc.description = (
    "Diffusion-model generated climate "
    "realizations"
)

nc.checkpoint = CHECKPOINT

nc.sample_steps = SAMPLE_STEPS

nc.samples_per_year = N_SAMPLES

nc.historical_years = (
    "1850-2014"
)

nc.ssp370_years = (
    "2015-2100"
)


# ============================================================
# GENERATE HISTORICAL
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "GENERATING HISTORICAL: 1850-2014"
)

print(
    "=" * 60
)


for year in SCENARIO_YEARS["hist"]:

    year = int(year)

    # --------------------------------------------------------
    # Find dataset index for this year
    # --------------------------------------------------------

    indices = np.where(
        hist_available_years == year
    )[0]

    if len(indices) != 1:

        raise RuntimeError(
            f"Could not uniquely locate "
            f"historical year {year}."
        )

    index = int(
        indices[0]
    )


    # --------------------------------------------------------
    # Get conditioning
    # --------------------------------------------------------

    x, cond = hist_dataset[
        index
    ]

    cond = cond.unsqueeze(0)

    cond = cond.to(
        device=device,
        dtype=torch.float32,
    )


    print(
        f"\nHistorical {year} "
        f"(index {index})"
    )


    # --------------------------------------------------------
    # Generate 10 samples simultaneously
    # --------------------------------------------------------

    generated = generate_samples(
        cond,
        N_SAMPLES,
    )


    generated = generated.numpy()


    # --------------------------------------------------------
    # Denormalize
    # --------------------------------------------------------

    trefht = (
        generated[:, 0]
        * 21.0
        + 4.5
    )

    prect = np.expm1(
        generated[:, 1]
        * 0.5703
        + 1.0727
    )


    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    year_index = (
        year - 1850
    )

    trefht_var[
        year_index,
        :,
        :,
        :
    ] = trefht[:, 0]

    prect_var[
        year_index,
        :,
        :,
        :
    ] = prect[:, 0]


    nc.sync()


# ============================================================
# GENERATE SSP370
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "GENERATING SSP370: 2015-2100"
)

print(
    "=" * 60
)


for year in SCENARIO_YEARS["ssp370"]:

    year = int(year)

    # --------------------------------------------------------
    # Find dataset index for this year
    # --------------------------------------------------------

    indices = np.where(
        ssp370_available_years == year
    )[0]

    if len(indices) != 1:

        raise RuntimeError(
            f"Could not uniquely locate "
            f"ssp370 year {year}."
        )

    index = int(
        indices[0]
    )


    # --------------------------------------------------------
    # Get conditioning
    # --------------------------------------------------------

    x, cond = ssp370_dataset[
        index
    ]

    cond = cond.unsqueeze(0)

    cond = cond.to(
        device=device,
        dtype=torch.float32,
    )


    print(
        f"\nSSP370 {year} "
        f"(index {index})"
    )


    # --------------------------------------------------------
    # Generate 10 samples simultaneously
    # --------------------------------------------------------

    generated = generate_samples(
        cond,
        N_SAMPLES,
    )


    generated = generated.numpy()


    # --------------------------------------------------------
    # Denormalize
    # --------------------------------------------------------

    trefht = (
        generated[:, 0]
        * 21.0
        + 4.5
    )

    prect = np.expm1(
        generated[:, 1]
        * 0.5703
        + 1.0727
    )


    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    year_index = (
        year - 1850
    )

    trefht_var[
        year_index,
        :,
        :,
        :
    ] = trefht[:, 0]

    prect_var[
        year_index,
        :,
        :,
        :
    ] = prect[:, 0]


    nc.sync()


# ============================================================
# CLOSE FILE
# ============================================================

nc.close()


# ============================================================
# FINISHED
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "DONE"
)

print(
    "=" * 60
)

print(
    "Output:",
    OUTPUT
)

print(
    "Years:",
    "1850-2100"
)

print(
    "Historical:",
    "1850-2014"
)

print(
    "SSP370:",
    "2015-2100"
)

print(
    "Samples per year:",
    N_SAMPLES
)

print(
    "Variables:",
    "TREFHT, PRECT"
)
