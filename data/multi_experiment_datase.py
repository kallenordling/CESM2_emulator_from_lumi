"""
multi_experiment_dataset.py
===========================
Multi-experiment wrapper around ClimateDataset.

Design goals
------------
* Wrap N ClimateDataset instances (one per SSP/historical scenario).
* Yield batches that contain samples from **multiple scenarios** so that
  the conditioning encoder receives strong contrastive gradients.
* Return a scalar ``scenario_id`` tensor alongside (x, cond) so the
  trainer can optionally compute a cross-scenario conditioning loss.
* Stay fully backward-compatible: a single-experiment run can still use
  the original ClimateDataset / ClimateDataLoader without changes.

Typical usage
-------------
    from climate_dataset import ClimateDataset
    from multi_experiment_dataset import (
        MultiExperimentDataset,
        MultiExperimentDataLoader,
    )

    ssp370 = ClimateDataset(seq_len=10, realizations=[...], data_dir="...",
                            target_vars=["TREFHT"], cond_file="co2_so2_ssp370.nc",
                            cond_vars=["CO2", "SUL"])
    ssp245 = ClimateDataset(seq_len=10, realizations=[...], data_dir="...",
                            target_vars=["TREFHT"], cond_file="co2_so2_ssp245.nc",
                            cond_vars=["CO2", "SUL"])

    multi_ds = MultiExperimentDataset([ssp370, ssp245],
                                       scenario_names=["ssp370", "ssp245"])
    loader   = MultiExperimentDataLoader(multi_ds, accelerator, batch_size=8)

    for epoch in range(n_epochs):
        for x, cond, scenario_ids in loader.generate():
            ...   # x: [B,V,T,H,W]  cond: [B,C,T,H,W]  ids: [B] int
"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from accelerate import Accelerator

from climate_dataset import ClimateDataset


# ---------------------------------------------------------------------------
# Flat index helpers
# ---------------------------------------------------------------------------

class _ExperimentIndex:
    """Maps a flat integer index to (experiment_idx, window_idx).

    The flat space is rebuilt whenever a realization changes inside any
    child dataset, so it always reflects the currently loaded data.
    """

    def __init__(self, datasets: list[ClimateDataset]):
        self._datasets = datasets
        self._lengths: list[int] = []
        self._offsets: list[int] = []
        self._total: int = 0
        self._rebuild()

    def _rebuild(self):
        self._lengths = [len(ds) for ds in self._datasets]
        self._offsets = []
        offset = 0
        for n in self._lengths:
            self._offsets.append(offset)
            offset += n
        self._total = offset

    def __len__(self) -> int:
        return self._total

    def decode(self, flat_idx: int) -> tuple[int, int]:
        """Return (experiment_idx, window_idx) for a flat index."""
        for exp_idx in range(len(self._datasets) - 1, -1, -1):
            if flat_idx >= self._offsets[exp_idx]:
                return exp_idx, flat_idx - self._offsets[exp_idx]
        raise IndexError(f"flat_idx={flat_idx} out of range [0, {self._total})")

    def encode(self, exp_idx: int, window_idx: int) -> int:
        return self._offsets[exp_idx] + window_idx

    def flat_indices_for_experiment(self, exp_idx: int) -> np.ndarray:
        n = self._lengths[exp_idx]
        return np.arange(self._offsets[exp_idx], self._offsets[exp_idx] + n)


# ---------------------------------------------------------------------------
# MultiExperimentDataset
# ---------------------------------------------------------------------------

class MultiExperimentDataset(Dataset):
    """Wraps multiple ClimateDataset instances as a single dataset.

    Each item returns ``(x, cond, scenario_id)`` where ``scenario_id`` is
    an integer tensor so the trainer can use it for contrastive losses or
    logging.

    Parameters
    ----------
    datasets:
        List of already-constructed ClimateDataset instances, one per
        experiment / scenario.  They must share the same ``seq_len``,
        ``target_vars``, and ``cond_vars``.
    scenario_names:
        Optional human-readable labels.  Defaults to "exp_0", "exp_1", ...
    """

    def __init__(
        self,
        datasets: list[ClimateDataset],
        scenario_names: Optional[list[str]] = None,
    ):
        if len(datasets) == 0:
            raise ValueError("Need at least one ClimateDataset.")

        # Validate compatibility
        ref = datasets[0]
        for i, ds in enumerate(datasets[1:], 1):
            if ds.seq_len != ref.seq_len:
                raise ValueError(
                    f"Dataset {i} has seq_len={ds.seq_len} but dataset 0 "
                    f"has seq_len={ref.seq_len}.  All datasets must match."
                )
            if ds.vars != ref.vars:
                raise ValueError(
                    f"Dataset {i} has target_vars={ds.vars} but dataset 0 "
                    f"has {ref.vars}.  All datasets must match."
                )
            if ds.cond_vars != ref.cond_vars:
                raise ValueError(
                    f"Dataset {i} has cond_vars={ds.cond_vars} but dataset 0 "
                    f"has {ref.cond_vars}.  All datasets must match."
                )

        self.datasets = datasets
        self.scenario_names: list[str] = (
            scenario_names
            if scenario_names is not None
            else [f"exp_{i}" for i in range(len(datasets))]
        )
        self._index = _ExperimentIndex(datasets)

        print(
            f"[MULTI] Initialised with {len(datasets)} experiments: "
            + ", ".join(
                f"{name}({len(ds)} windows)"
                for name, ds in zip(self.scenario_names, datasets)
            )
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, flat_idx: int):
        exp_idx, window_idx = self._index.decode(flat_idx)
        x, cond = self.datasets[exp_idx][window_idx]
        scenario_id = torch.tensor(exp_idx, dtype=torch.long)
        return x, cond, scenario_id

    # ------------------------------------------------------------------
    # Realization management
    # ------------------------------------------------------------------

    def load_realization(self, exp_idx: int, realization: str):
        """Load a new realization for experiment ``exp_idx`` and rebuild index."""
        self.datasets[exp_idx].load_data(realization)
        self._index._rebuild()
        print(
            f"[MULTI] Loaded realization '{realization}' for "
            f"experiment '{self.scenario_names[exp_idx]}'  "
            f"total_windows={len(self._index)}"
        )

    def estimate_num_batches(self, batch_size: int) -> int:
        """Rough estimate across all experiments and realizations."""
        total = sum(
            ds.estimate_num_batches(batch_size)
            for ds in self.datasets
        )
        return total

    @property
    def seq_len(self) -> int:
        return self.datasets[0].seq_len

    @property
    def n_experiments(self) -> int:
        return len(self.datasets)


# ---------------------------------------------------------------------------
# MultiExperimentDataLoader
# ---------------------------------------------------------------------------

class MultiExperimentDataLoader:
    """Iterates over all realizations of all experiments, mixing scenarios.

    Batch composition strategy
    --------------------------
    ``mix_scenarios=True``  (default, recommended):
        Each batch is assembled by drawing ``batch_size // n_experiments``
        windows from each scenario.  This guarantees every batch sees
        the full range of emission trajectories, providing a strong
        contrastive conditioning signal.

    ``mix_scenarios=False``:
        Falls back to plain per-realization sequential batches (equivalent
        to the original ClimateDataLoader used N times).  Useful for
        baselines or when you want to measure per-scenario loss.

    Realization scheduling
    ----------------------
    The loader iterates over a "realization plan": for each experiment it
    produces a shuffled list of realizations.  In each step of the plan the
    corresponding experiment dataset loads one realization, then batches are
    drawn from the current multi-experiment window pool.

    Parameters
    ----------
    dataset:
        A MultiExperimentDataset.
    accelerator:
        HuggingFace Accelerate Accelerator.
    batch_size:
        Total samples per batch (split equally across scenarios when
        ``mix_scenarios=True``; rounded down to a multiple of
        ``n_experiments``).
    mix_scenarios:
        Whether to guarantee cross-scenario batches.  Default: True.
    steps_per_realization:
        How many batches to draw from the current realization window pool
        before rotating to the next realization.  ``None`` means use all
        available batches.
    **dataloader_kwargs:
        Forwarded to ``torch.utils.data.DataLoader`` in non-mixed mode.
    """

    def __init__(
        self,
        dataset: MultiExperimentDataset,
        accelerator: Accelerator,
        batch_size: int,
        mix_scenarios: bool = True,
        steps_per_realization: Optional[int] = None,
        **dataloader_kwargs: Any,
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.mix_scenarios = mix_scenarios
        self.steps_per_realization = steps_per_realization
        self.dataloader_kwargs = dataloader_kwargs
        self.n_exp = dataset.n_experiments

        if mix_scenarios and batch_size % self.n_exp != 0:
            batch_size = (batch_size // self.n_exp) * self.n_exp
            print(
                f"[MULTI] batch_size rounded down to {batch_size} "
                f"(must be divisible by n_experiments={self.n_exp})"
            )
        self.batch_size = batch_size
        self.per_exp = batch_size // self.n_exp if mix_scenarios else batch_size

        print(
            f"[MULTI] DataLoader ready  "
            f"batch_size={batch_size}  "
            f"mix_scenarios={mix_scenarios}  "
            f"per_exp={self.per_exp}  "
            f"n_exp={self.n_exp}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.dataset.estimate_num_batches(self.batch_size)

    # ------------------------------------------------------------------

    def _all_realization_plans(self) -> list[list[str]]:
        """Return shuffled realization list for each experiment."""
        plans = []
        for ds in self.dataset.datasets:
            r = list(ds.realizations)
            random.shuffle(r)
            plans.append(r)
        return plans

    # ------------------------------------------------------------------

    def generate(self):
        """Yield ``(x, cond, scenario_ids)`` batches across all realizations.

        Yields
        ------
        x:            torch.Tensor  [B, V, T, H, W]
        cond:         torch.Tensor  [B, C, T, H, W]
        scenario_ids: torch.Tensor  [B]  (long)
        """
        plans = self._all_realization_plans()
        # Max realizations across experiments; shorter lists are cycled.
        max_r = max(len(p) for p in plans)

        for r_step in range(max_r):
            # Load one realization per experiment for this step
            for exp_idx, plan in enumerate(plans):
                realization = plan[r_step % len(plan)]
                self.dataset.load_realization(exp_idx, realization)

            if self.mix_scenarios:
                yield from self._generate_mixed()
            else:
                yield from self._generate_sequential()

    # ------------------------------------------------------------------

    def _generate_mixed(self):
        """Yield cross-scenario batches from the current window pool."""
        # Build per-experiment window index arrays
        index_pools: list[np.ndarray] = []
        for exp_idx in range(self.n_exp):
            flat_idx = self.dataset._index.flat_indices_for_experiment(exp_idx)
            shuffled = np.random.permutation(flat_idx)
            index_pools.append(shuffled)

        # Number of full batches is limited by the smallest pool
        n_batches = min(len(pool) for pool in index_pools) // self.per_exp
        if self.steps_per_realization is not None:
            n_batches = min(n_batches, self.steps_per_realization)

        if n_batches == 0:
            print(
                f"[MULTI] WARNING: not enough windows for a full mixed batch "
                f"(per_exp={self.per_exp}).  Skipping realization step."
            )
            return

        device = self.accelerator.device

        for b in range(n_batches):
            s, e = b * self.per_exp, (b + 1) * self.per_exp
            batch_flat: list[int] = []
            for pool in index_pools:
                batch_flat.extend(pool[s:e].tolist())

            # Shuffle so scenarios are interleaved within the batch
            random.shuffle(batch_flat)

            samples = [self.dataset[i] for i in batch_flat]
            x_batch    = torch.stack([s[0] for s in samples]).to(device)
            cond_batch = torch.stack([s[1] for s in samples]).to(device)
            ids_batch  = torch.stack([s[2] for s in samples]).to(device)

            yield x_batch, cond_batch, ids_batch

    # ------------------------------------------------------------------

    def _generate_sequential(self):
        """Yield per-experiment batches (non-mixed fallback).

        Iterates experiment by experiment, standard DataLoader per realization.
        """
        device = self.accelerator.device

        for exp_idx, ds in enumerate(self.dataset.datasets):
            flat_indices = self.dataset._index.flat_indices_for_experiment(exp_idx)
            n = len(flat_indices)
            if n == 0:
                continue

            shuffled = np.random.permutation(flat_indices)
            n_batches = n // self.batch_size
            if self.steps_per_realization is not None:
                n_batches = min(n_batches, self.steps_per_realization)

            for b in range(n_batches):
                s, e = b * self.batch_size, (b + 1) * self.batch_size
                batch_flat = shuffled[s:e].tolist()

                samples = [self.dataset[i] for i in batch_flat]
                x_batch    = torch.stack([s[0] for s in samples]).to(device)
                cond_batch = torch.stack([s[1] for s in samples]).to(device)
                ids_batch  = torch.stack([s[2] for s in samples]).to(device)

                yield x_batch, cond_batch, ids_batch


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_multi_experiment_loader(
    experiment_configs: list[dict],
    accelerator: Accelerator,
    batch_size: int,
    mix_scenarios: bool = True,
    steps_per_realization: Optional[int] = None,
    **shared_dataset_kwargs: Any,
) -> MultiExperimentDataLoader:
    """Convenience factory: build datasets from a list of config dicts.

    Each entry in ``experiment_configs`` is a dict that can override any
    ClimateDataset keyword argument.  Keys shared across all experiments
    can be passed as ``**shared_dataset_kwargs``.

    Example
    -------
    ::

        loader = build_multi_experiment_loader(
            experiment_configs=[
                {"cond_file": "emissions_ssp370.nc", "data_dir": "/data/ssp370",
                 "realizations": ["r1i1p1f1", "r2i1p1f1"],
                 "scenario_name": "ssp370"},
                {"cond_file": "emissions_ssp245.nc", "data_dir": "/data/ssp245",
                 "realizations": ["r1i1p1f1"],
                 "scenario_name": "ssp245"},
                {"cond_file": "emissions_ssp585.nc", "data_dir": "/data/ssp585",
                 "realizations": ["r1i1p1f1"],
                 "scenario_name": "ssp585"},
            ],
            accelerator=accelerator,
            batch_size=12,
            # shared kwargs
            seq_len=10,
            target_vars=["TREFHT"],
            cond_vars=["CO2", "SUL"],
        )
    """
    datasets: list[ClimateDataset] = []
    scenario_names: list[str] = []

    for cfg in experiment_configs:
        cfg = dict(cfg)  # shallow copy so we don't mutate caller's dict
        name = cfg.pop("scenario_name", None)
        # Merge shared kwargs (cfg takes priority)
        merged = {**shared_dataset_kwargs, **cfg}
        datasets.append(ClimateDataset(**merged))
        scenario_names.append(name or f"exp_{len(datasets) - 1}")

    multi_ds = MultiExperimentDataset(datasets, scenario_names=scenario_names)
    return MultiExperimentDataLoader(
        multi_ds,
        accelerator,
        batch_size=batch_size,
        mix_scenarios=mix_scenarios,
        steps_per_realization=steps_per_realization,
    )