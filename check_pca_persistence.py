"""Cheap pre-check for the PCA-persistence fix (run on LUMI where the cond
files are reachable; the local box has no /scratch mount).

Builds the aaer conditioning tensor two ways and diffs the SUL channel at 2015:
  (a) pca_objects=None  → build_cond_tensor FITS a fresh 5-EOF SUL basis
  (b) a persisted basis → build_cond_tensor APPLIES that basis

(a) is what eval did before this fix when ckpt["PCA"] was absent (it actually
passed pca_cond=None → NO projection at all, so SUL was raw/un-PCA'd). (b) is
what training fed the model. A large SUL diff at 2015 quantifies the train↔eval
mismatch the persisted basis removes.

Usage (on LUMI):
    python check_pca_persistence.py
"""
import numpy as np
from omegaconf import OmegaConf

from eval_aero import build_cond_tensor, EMIS_DIR
import os

COND_VARS = ["CO2", "SUL"]
TIME_DIM = "time"
AAER_COND = os.path.join(EMIS_DIR, "emissions_aaer_only_timefixed.nc")

data_cfg = OmegaConf.load("configs/config_data.yaml")
N_COMP = OmegaConf.to_container(data_cfg.n_components_cond, resolve=True)  # [30, 5]
CS = OmegaConf.to_container(data_cfg.cond_smooth_sigma, resolve=True)      # [0, 2]
CSM = data_cfg.get("cond_smooth_method", "gaussian")

# (a) raw eval-before-fix: no PCA projection at all (pca_objects=None)
cond_raw, years, _, _ = build_cond_tensor(
    AAER_COND, COND_VARS, TIME_DIM, None, N_COMP, CS, CSM)

# Fit a fresh 5-EOF basis the way a child ClimateDataset does, then APPLY it —
# this is the basis training fed the model and that the fix now persists.
from data.climate_dataset import pca_denoise_dataset
# Re-derive the smoothed-but-unprojected tensor, fit, then reapply.
_, fitted = pca_denoise_dataset(cond_raw.clone(), n_components=N_COMP,
                                var_names=COND_VARS, pca_objects=None)
cond_pca, _ = build_cond_tensor(
    AAER_COND, COND_VARS, TIME_DIM, fitted, N_COMP, CS, CSM)

t2015 = int(np.where(years == 2015)[0][0])
sul_raw = cond_raw[1, t2015].numpy()
sul_pca = cond_pca[1, t2015].numpy()
d = sul_pca - sul_raw
print(f"SUL @2015  raw[min/mean/max]={sul_raw.min():.3f}/{sul_raw.mean():.3f}/{sul_raw.max():.3f}")
print(f"SUL @2015  pca[min/mean/max]={sul_pca.min():.3f}/{sul_pca.mean():.3f}/{sul_pca.max():.3f}")
print(f"SUL @2015  |diff| mean={np.abs(d).mean():.4f}  max={np.abs(d).max():.4f}  "
      f"RMS={np.sqrt((d**2).mean()):.4f}")
print("Non-trivial diff => persisting the basis (this fix) changes what eval feeds the model.")
