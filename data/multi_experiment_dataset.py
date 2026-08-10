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
    from climate_dataset import ClimateDataset, EvalClimateDataset
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

import queue
import random
import threading
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from accelerate import Accelerator

from data.climate_dataset import ClimateDataset, EvalClimateDataset, StratifiedPeriodSampler


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

        #print(
        #    f"[MULTI] Initialised with {len(datasets)} experiments: "
        #    + ", ".join(
        #        f"{name}({len(ds)} windows)"
        #        for name, ds in zip(self.scenario_names, datasets)
        #    )
        #)

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
        #print(
        #    f"[MULTI] Loaded realization '{realization}' for "
        #    f"experiment '{self.scenario_names[exp_idx]}'  "
        #    f"total_windows={len(self._index)}"
        #)

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

    # ------------------------------------------------------------------
    # PCA persistence (per-scenario)
    # ------------------------------------------------------------------

    def get_pca_state(self) -> dict:
        """Aggregate the per-scenario fitted PCA bases for checkpointing.

        Each child ClimateDataset fits its OWN PCA basis (cond + target) on
        its first ``load_data``. eval_aero.py applies a single ``pca_cond`` to
        whichever scenario's cond_file it loads, so we persist:

          * top-level ``"cond"`` / ``"target"`` — a single reference basis kept
            for backward compatibility with the flat ``ckpt["PCA"]`` that eval
            consumes (eval_aero.py:1656). The reference is the ``aaer`` scenario
            when present (this is the channel whose train↔eval basis mismatch
            speckled the maps), else the first scenario.
          * ``"per_scenario"`` — ``{scenario_name: {"cond": [...], "target": [...]}}``
            so eval can select the basis matching the scenario being evaluated
            instead of forcing one basis onto all of them.
        """
        per_scenario = {
            name: ds.get_pca_state()
            for name, ds in zip(self.scenario_names, self.datasets)
        }
        # Pick the reference basis used by flat-key consumers.
        ref_name = next(
            (n for n in self.scenario_names if "aaer" in n.lower()),
            self.scenario_names[0],
        )
        ref = per_scenario[ref_name]
        return {
            "cond":         ref.get("cond"),
            "target":       ref.get("target"),
            "ref_scenario": ref_name,
            "per_scenario": per_scenario,
        }

    def set_pca_state(self, state: dict) -> None:
        """Restore per-scenario PCA bases saved by :meth:`get_pca_state`.

        Restores each child dataset from the matching entry in
        ``state["per_scenario"]`` so a resumed run keeps the basis it was
        trained with instead of re-fitting on the first realization. Falls
        back to the flat reference basis for scenarios missing from the map
        (e.g. a checkpoint saved before per-scenario persistence) so the
        restore degrades gracefully rather than silently re-fitting.
        """
        if state is None:
            return
        per_scenario = state.get("per_scenario") or {}
        flat = {"cond": state.get("cond"), "target": state.get("target")}
        for name, ds in zip(self.scenario_names, self.datasets):
            ds.set_pca_state(per_scenario.get(name, flat))


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
        stratified: bool = False,
        period_boundaries: tuple = (1950, 2020),
        steps_per_realization: Optional[int] = None,
        scenario_weights: Optional[list] = None,
        prefetch_batches: int = 2,
        year_bias: float = 0.0,
        year_bias_floor: float = 0.05,
        bsp_depth: int = 0,
        shard_across_ranks: bool = True,
        **dataloader_kwargs: Any,
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.mix_scenarios = mix_scenarios
        self.stratified = stratified
        self.period_boundaries = period_boundaries
        self.steps_per_realization = steps_per_realization
        self.dataloader_kwargs = dataloader_kwargs
        self.n_exp = dataset.n_experiments
        self.prefetch_batches = prefetch_batches
        self.year_bias = float(year_bias)
        self.year_bias_floor = float(year_bias_floor)
        self.bsp_depth = int(bsp_depth)
        self.shard_across_ranks = bool(shard_across_ranks)
        if self.year_bias != 0.0:
            global_min = float("inf")
            global_max = float("-inf")
            for exp_idx in range(self.n_exp):
                n_win = len(self.dataset._index.flat_indices_for_experiment(exp_idx))
                if n_win == 0:
                    continue
                years = self.dataset.datasets[exp_idx]._time_values[:n_win].astype(np.float64)
                global_min = min(global_min, float(years.min()))
                global_max = max(global_max, float(years.max()))
            self._global_y_min = global_min
            self._global_y_max = global_max
            print(
                f"[MULTI] Year-biased sampling enabled  "
                f"year_bias={self.year_bias}  floor={self.year_bias_floor} "
                f"(weight ∝ ((year-{global_min:.0f})/({global_max:.0f}-{global_min:.0f}) "
                f"+ floor)^year_bias, global axis across experiments)"
            )

        if self.bsp_depth > 0:
            n_buckets = 1 << self.bsp_depth
            global_min = float("inf")
            global_max = float("-inf")
            for exp_idx in range(self.n_exp):
                n_win = len(self.dataset._index.flat_indices_for_experiment(exp_idx))
                if n_win == 0:
                    continue
                years = self.dataset.datasets[exp_idx]._time_values[:n_win].astype(np.float64)
                global_min = min(global_min, float(years.min()))
                global_max = max(global_max, float(years.max()))
            self._bsp_y_min = global_min
            self._bsp_y_max = global_max
            span = max(global_max - global_min, 1.0)
            # Per-experiment list of bucket arrays (each is a flat-index array).
            self._bsp_buckets: list[list[np.ndarray]] = []
            bucket_summary: list[str] = []
            for exp_idx in range(self.n_exp):
                flat_idx = self.dataset._index.flat_indices_for_experiment(exp_idx)
                n_win = len(flat_idx)
                if n_win == 0:
                    self._bsp_buckets.append([])
                    continue
                years = self.dataset.datasets[exp_idx]._time_values[:n_win].astype(np.float64)
                bidx = np.floor((years - global_min) / span * n_buckets).astype(int)
                bidx = np.clip(bidx, 0, n_buckets - 1)
                buckets: list[np.ndarray] = []
                for b in range(n_buckets):
                    sel = flat_idx[bidx == b]
                    if len(sel) > 0:
                        buckets.append(sel)
                self._bsp_buckets.append(buckets)
                name = self.dataset.scenario_names[exp_idx] if hasattr(self.dataset, "scenario_names") else f"exp{exp_idx}"
                bucket_summary.append(f"{name}={len(buckets)}/{n_buckets}")
            print(
                f"[MULTI] BSP sampling enabled  bsp_depth={self.bsp_depth} "
                f"({n_buckets} buckets over {global_min:.0f}-{global_max:.0f}); "
                f"non-empty per exp: {', '.join(bucket_summary)}"
            )

        if mix_scenarios and scenario_weights is not None:
            # Convert weights to per-experiment integer sample counts that sum
            # to batch_size (largest-remainder rounding to avoid drift).
            weights = np.array(scenario_weights, dtype=float)
            if len(weights) != self.n_exp:
                raise ValueError(
                    f"scenario_weights has {len(weights)} entries but there "
                    f"are {self.n_exp} experiments."
                )
            weights = weights / weights.sum()
            raw   = weights * batch_size
            counts = np.floor(raw).astype(int)
            for idx in np.argsort(-(raw - counts))[:batch_size - counts.sum()]:
                counts[idx] += 1
            self.per_exp_list = counts.tolist()
            self.batch_size   = batch_size
            self.per_exp      = max(counts)
            print(
                "[MULTI] scenario_weights → samples per batch: "
                + ", ".join(
                    f"{name}={n}"
                    for name, n in zip(dataset.scenario_names, self.per_exp_list)
                )
            )
        else:
            if mix_scenarios and batch_size % self.n_exp != 0:
                batch_size = (batch_size // self.n_exp) * self.n_exp
            self.batch_size   = batch_size
            self.per_exp      = batch_size // self.n_exp if mix_scenarios else batch_size
            self.per_exp_list = [self.per_exp] * self.n_exp

        if stratified:
            print(
                f"[MULTI] Stratified period sampling enabled  "
                f"boundaries={period_boundaries}"
            )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.dataset.estimate_num_batches(self.batch_size)

    # ------------------------------------------------------------------

    def _rank_batch_range(self, n_batches: int) -> tuple[int, int]:
        """Per-rank [start, end) over the global batch index range.

        With shared RNG seed (`set_seed(..., device_specific=False)` in
        main_aero.py) every rank generates an identical index pool. Slicing
        by rank therefore gives each rank a disjoint, non-overlapping slab
        of batches per epoch, restoring real DDP weak scaling.

        Floor division — discard remainder so all ranks do the same number
        of optimizer steps (otherwise DDP allreduce would hang).
        """
        # One-time diagnostic: log once per rank what _rank_batch_range
        # actually does, including which (if any) early-return fires.
        ws = self.accelerator.num_processes if self.accelerator is not None else 0
        rank = self.accelerator.process_index if self.accelerator is not None else -1
        if not getattr(self, "_logged_shard", False):
            print(
                f"[DDP-LOADER] rank={rank}/{ws} n_batches={n_batches} "
                f"shard_across_ranks={self.shard_across_ranks} "
                f"accelerator={'set' if self.accelerator is not None else 'None'}",
                flush=True,
            )
            self._logged_shard = True

        if not self.shard_across_ranks or self.accelerator is None:
            return 0, n_batches
        if ws <= 1 or n_batches < ws:
            return 0, n_batches
        per_rank = n_batches // ws
        start = rank * per_rank
        end = start + per_rank
        return start, end

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

        When ``prefetch_batches > 0`` a background thread assembles the next
        batch on CPU while the GPU processes the current one, then the main
        thread transfers to GPU with ``non_blocking=True``.
        """
        if self.prefetch_batches > 0:
            yield from self._prefetch_generate()
        else:
            yield from self._generate_epoch()

    def _generate_epoch(self):
        """Core epoch loop — loads realizations and yields batches to device."""
        plans = self._all_realization_plans()
        max_r = max(len(p) for p in plans)
        device = self.accelerator.device

        for r_step in range(max_r):
            for exp_idx, plan in enumerate(plans):
                realization = plan[r_step % len(plan)]
                self.dataset.load_realization(exp_idx, realization)

            if self.mix_scenarios:
                yield from self._generate_mixed()
            else:
                yield from self._generate_sequential()

    def _prefetch_generate(self):
        """Wraps _generate_epoch with a background-thread prefetch queue.

        The producer thread assembles batches on CPU (tensor indexing only,
        no CUDA calls).  The main thread pops CPU batches from the queue and
        transfers to GPU with non_blocking=True, overlapping the next batch
        assembly with the current GPU forward/backward pass.
        """
        device = self.accelerator.device
        q: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        _sentinel = object()

        def _producer():
            try:
                plans = self._all_realization_plans()
                max_r = max(len(p) for p in plans)
                for r_step in range(max_r):
                    for exp_idx, plan in enumerate(plans):
                        self.dataset.load_realization(exp_idx, plan[r_step % len(plan)])
                    # Assemble batches on CPU, no GPU transfer yet
                    gen = (
                        self._generate_mixed_cpu()
                        if self.mix_scenarios
                        else self._generate_sequential()
                    )
                    for batch in gen:
                        q.put(batch)
            except Exception as e:
                q.put(e)
            finally:
                q.put(_sentinel)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is _sentinel:
                break
            if isinstance(item, Exception):
                raise item
            x, cond, ids = item
            yield (
                x.to(device, non_blocking=True),
                cond.to(device, non_blocking=True),
                ids.to(device, non_blocking=True),
            )

    def _generate_mixed_cpu(self):
        """Like _generate_mixed but returns CPU tensors (for prefetch thread)."""
        if self.bsp_depth > 0:
            yield from self._generate_mixed_bsp(device=None, to_device=False)
        elif self.stratified:
            yield from self._generate_mixed_stratified(device=None, to_device=False)
        else:
            yield from self._generate_mixed_uniform(device=None, to_device=False)

    # ------------------------------------------------------------------

    def _generate_mixed(self):
        """Yield cross-scenario batches from the current window pool.

        When ``stratified=True`` each batch is assembled so that every
        period (historical / present / future) is represented equally
        across all experiments.  Concretely, for n_exp=4 and per_exp=4:

            batch = [hist_exp0, hist_exp1, hist_exp2, hist_exp3,
                     pres_exp0, pres_exp1, pres_exp2, pres_exp3,
                     fut_exp0,  fut_exp1,  fut_exp2,  fut_exp3]  (shuffled)

        This guarantees the conditioning encoder sees large emission
        contrasts (across both scenarios AND time periods) in every step.

        When ``stratified=False`` (default) the original uniform-random
        behaviour is preserved.
        """
        device = self.accelerator.device

        if self.bsp_depth > 0:
            yield from self._generate_mixed_bsp(device)
        elif self.stratified:
            yield from self._generate_mixed_stratified(device)
        else:
            yield from self._generate_mixed_uniform(device)

    def _vectorized_fetch(
        self,
        window_indices_per_exp: list[np.ndarray],
        device,
        shuffle: bool = True,
        to_device: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble a batch via vectorized tensor indexing (no Python loop per sample).

        Parameters
        ----------
        window_indices_per_exp:
            List of length n_exp; each element is an int array of local
            window indices (0-based within that experiment's tensor_data).
        device:
            Target device.
        shuffle:
            Whether to randomly permute the assembled batch.
        """
        seq_len = self.dataset.seq_len
        t_offsets = torch.arange(seq_len, dtype=torch.long)
        x_parts: list[torch.Tensor] = []
        cond_parts: list[torch.Tensor] = []
        ids_parts: list[torch.Tensor] = []

        for exp_idx, win_idx in enumerate(window_indices_per_exp):
            ds = self.dataset.datasets[exp_idx]
            # t_idx: [B_exp, seq_len]
            t_idx = torch.from_numpy(win_idx.astype(np.int64)).unsqueeze(1) + t_offsets
            # tensor_data: [V, T, H, W] → indexed by t_idx → [V, B_exp, seq_len, H, W]
            x_parts.append(ds.tensor_data[:, t_idx].permute(1, 0, 2, 3, 4))
            cond_parts.append(ds.tensor_data_cond[:, t_idx].permute(1, 0, 2, 3, 4))
            ids_parts.append(torch.full((len(win_idx),), exp_idx, dtype=torch.long))

        x_cat    = torch.cat(x_parts,    dim=0)
        cond_cat = torch.cat(cond_parts, dim=0)
        ids_cat  = torch.cat(ids_parts,  dim=0)

        if shuffle:
            perm = torch.randperm(x_cat.shape[0])
            x_cat, cond_cat, ids_cat = x_cat[perm], cond_cat[perm], ids_cat[perm]

        if not to_device:
            return x_cat, cond_cat, ids_cat

        return (
            x_cat.to(device, non_blocking=True),
            cond_cat.to(device, non_blocking=True),
            ids_cat.to(device, non_blocking=True),
        )

    def _generate_mixed_uniform(self, device, to_device: bool = True):
        """Uniform-random mixed batches, respecting per-experiment sample counts.

        When ``self.year_bias > 0`` windows are drawn (with replacement) using
        per-window probabilities ∝ ((year - y_min)/(y_max - y_min) + floor)^year_bias
        with y_min/y_max taken from the *global* year range across all
        experiments — so a year close to a scenario boundary (e.g. hist 2014
        vs ssp370 2015) gets nearly identical weight in either experiment.
        """
        offsets = self.dataset._index._offsets
        # Scale draw size with world_size so n_batches ≥ ws and the per-rank
        # sharding path in _rank_batch_range stays active under 4+ nodes.
        # Pool is drawn with replacement (year-biased) or by tiling permutations
        # (uniform), so expanding it costs only a bit of RAM and gives each rank
        # disjoint work instead of the unsharded fallback.
        ws = self.accelerator.num_processes if self.accelerator is not None else 1
        index_pools: list[np.ndarray] = []
        for exp_idx in range(self.n_exp):
            flat_idx = self.dataset._index.flat_indices_for_experiment(exp_idx)
            n_win = len(flat_idx)
            target_pool = max(n_win, ws * self.per_exp_list[exp_idx])
            if self.year_bias != 0.0 and n_win > 0:
                ds = self.dataset.datasets[exp_idx]
                years = ds._time_values[:n_win].astype(np.float64)
                span = max(self._global_y_max - self._global_y_min, 1.0)
                w = ((years - self._global_y_min) / span + self.year_bias_floor) ** self.year_bias
                w = w / w.sum()
                # Draw enough samples up front for the whole epoch (with replacement)
                local = np.random.choice(n_win, size=target_pool, replace=True, p=w)
                index_pools.append(flat_idx[local])
            elif n_win > 0:
                # Tile a permutation up to the target pool size so 4-node runs
                # still get ws disjoint batches instead of falling back to
                # unsharded. Each tile is a fresh permutation to avoid blocks
                # of identical batches.
                tiles = -(-target_pool // n_win)  # ceil
                parts = [np.random.permutation(flat_idx) for _ in range(tiles)]
                index_pools.append(np.concatenate(parts)[:target_pool])
            else:
                index_pools.append(flat_idx)

        n_batches = min(
            len(pool) // self.per_exp_list[i]
            for i, pool in enumerate(index_pools)
            if self.per_exp_list[i] > 0
        )
        if self.steps_per_realization is not None:
            n_batches = min(n_batches, self.steps_per_realization)
        if n_batches == 0:
            return

        b_start, b_end = self._rank_batch_range(n_batches)
        for b in range(b_start, b_end):
            window_indices_per_exp = [
                pool[b * self.per_exp_list[i] : (b + 1) * self.per_exp_list[i]]
                - offsets[i]
                for i, pool in enumerate(index_pools)
            ]
            yield self._vectorized_fetch(window_indices_per_exp, device, to_device=to_device)

    def _generate_mixed_bsp(self, device, to_device: bool = True):
        """BSP-stratified epoch pool.

        Per experiment: round-robin interleave its non-empty year buckets
        so every n_buckets consecutive samples cover all buckets once
        (with replacement within bucket when needed). Slicing per_exp[i]
        consecutive entries per batch then gives each batch a balanced
        spread across the experiment's year range, even when the bucket
        is small (e.g. 1850-1900 for hist).
        """
        offsets = self.dataset._index._offsets
        index_pools: list[np.ndarray] = []

        for exp_idx in range(self.n_exp):
            buckets = self._bsp_buckets[exp_idx]
            per_exp = self.per_exp_list[exp_idx]
            if per_exp == 0 or len(buckets) == 0:
                index_pools.append(np.empty(0, dtype=np.int64))
                continue

            n_b = len(buckets)
            # Aim to consume each bucket at most once before reshuffling.
            # rounds = ceil(largest_bucket / 1) but capped to keep epochs
            # comparable to uniform (~mean bucket size per round).
            largest = max(len(b) for b in buckets)
            # One "round" = one draw per bucket. After n_b rounds, each batch
            # of size per_exp has cycled through all buckets at least once.
            # Also enforce ws-aware floor so pool_len = rounds * n_b ≥ ws *
            # per_exp, which keeps n_batches ≥ ws and DDP sharding active.
            ws = self.accelerator.num_processes if self.accelerator is not None else 1
            ws_floor = -(-ws * per_exp // n_b)  # ceil(ws * per_exp / n_b)
            rounds = max(largest, per_exp * 4, ws_floor)
            shuffled: list[np.ndarray] = []
            for b in buckets:
                if len(b) >= rounds:
                    shuffled.append(np.random.permutation(b)[:rounds])
                else:
                    # Cycle with reshuffle to avoid identical repeats.
                    parts = []
                    needed = rounds
                    while needed > 0:
                        take = min(len(b), needed)
                        parts.append(np.random.permutation(b)[:take])
                        needed -= take
                    shuffled.append(np.concatenate(parts))
            # Interleave: pool[k*n_b + j] = shuffled[j][k]
            stacked = np.stack(shuffled, axis=1)  # [rounds, n_b]
            pool = stacked.reshape(-1)             # [rounds * n_b]
            index_pools.append(pool)

        n_batches = min(
            len(pool) // self.per_exp_list[i]
            for i, pool in enumerate(index_pools)
            if self.per_exp_list[i] > 0
        )
        if self.steps_per_realization is not None:
            n_batches = min(n_batches, self.steps_per_realization)
        if n_batches == 0:
            return

        b_start, b_end = self._rank_batch_range(n_batches)
        for b in range(b_start, b_end):
            window_indices_per_exp = [
                pool[b * self.per_exp_list[i] : (b + 1) * self.per_exp_list[i]]
                - offsets[i]
                for i, pool in enumerate(index_pools)
            ]
            yield self._vectorized_fetch(window_indices_per_exp, device, to_device=to_device)

    def _generate_mixed_stratified(self, device, to_device: bool = True):
        """Period-stratified mixed batches.

        For each experiment, build a StratifiedPeriodSampler and get its
        per-period index arrays.  Then zip them across experiments so that
        each batch slot [period, exp] draws one window — guaranteeing
        every batch sees all periods × all scenarios.

        Batch layout (per_period = per_exp // 3, n_exp experiments):
            [hist × n_exp | present × n_exp | future × n_exp]  shuffled
        """
        # per_exp must be divisible by 3 for equal period coverage
        per_exp = self.per_exp
        if per_exp % 3 != 0:
            per_exp = (per_exp // 3) * 3
            if per_exp == 0:
                print("[MULTI] WARNING: per_exp < 3, falling back to uniform sampling")
                yield from self._generate_mixed_uniform(device, to_device=to_device)
                return

        per_period_per_exp = per_exp // 3  # samples per (period, experiment) slot

        # Build one sampler per experiment using the currently loaded data
        samplers: list[StratifiedPeriodSampler] = []
        for exp_idx, ds in enumerate(self.dataset.datasets):
            samp = StratifiedPeriodSampler(
                ds,
                batch_size=per_exp,           # batch_size only used for per_period calc
                period_boundaries=self.period_boundaries,
                shuffle=True,
            )
            samp._build_indices()
            samplers.append(samp)

        # Check all experiments have all three periods
        if any(s._hist_idx is None for s in samplers):
            print("[MULTI] WARNING: some experiments missing a period — falling back to uniform")
            yield from self._generate_mixed_uniform(device, to_device=to_device)
            return

        # Offset arrays: sampler indices are local (window_idx), need flat
        offsets = [self.dataset._index._offsets[i] for i in range(self.n_exp)]

        def sample_period(arr: np.ndarray, n: int) -> np.ndarray:
            arr = np.random.permutation(arr)
            if len(arr) < n:
                arr = np.tile(arr, (n // len(arr) + 1))
            return arr[:n]

        # How many batches can we make? Limited by smallest period across all experiments
        n_batches = min(
            min(len(s._hist_idx), len(s._pres_idx), len(s._fut_idx))
            for s in samplers
        ) // per_period_per_exp

        if self.steps_per_realization is not None:
            n_batches = min(n_batches, self.steps_per_realization)

        if n_batches == 0:
            print("[MULTI] WARNING: not enough stratified windows — falling back to uniform")
            yield from self._generate_mixed_uniform(device, to_device=to_device)
            return

        # Pre-draw shuffled indices for all experiments and periods
        h_pools = [sample_period(s._hist_idx, n_batches * per_period_per_exp) for s in samplers]
        p_pools = [sample_period(s._pres_idx, n_batches * per_period_per_exp) for s in samplers]
        f_pools = [sample_period(s._fut_idx,  n_batches * per_period_per_exp) for s in samplers]

        b_start, b_end = self._rank_batch_range(n_batches)
        for b in range(b_start, b_end):
            sl = slice(b * per_period_per_exp, (b + 1) * per_period_per_exp)
            # Pools already contain local window indices (no offset needed)
            window_indices_per_exp = [
                np.concatenate([h_pools[exp_idx][sl], p_pools[exp_idx][sl], f_pools[exp_idx][sl]])
                for exp_idx in range(self.n_exp)
            ]
            yield self._vectorized_fetch(window_indices_per_exp, device, to_device=to_device)

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

            offset = self.dataset._index._offsets[exp_idx]
            b_start, b_end = self._rank_batch_range(n_batches)
            for b in range(b_start, b_end):
                s, e = b * self.batch_size, (b + 1) * self.batch_size
                window_indices = shuffled[s:e] - offset
                yield self._vectorized_fetch([window_indices], device, shuffle=False)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_multi_experiment_loader(
    experiment_configs: list[dict],
    accelerator: Accelerator,
    batch_size: int,
    mix_scenarios: bool = True,
    stratified: bool = False,
    period_boundaries: tuple = (1950, 2020),
    steps_per_realization: Optional[int] = None,
    scenario_weights: Optional[list] = None,
    prefetch_batches: int = 2,
    year_bias: float = 0.0,
    year_bias_floor: float = 0.05,
    bsp_depth: int = 0,
    shard_across_ranks: bool = True,
    all_years: bool = False,
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
    # Strip DataLoader-only kwargs so they don't leak into ClimateDataset.__init__().
    _DATALOADER_KEYS = {"num_workers", "pin_memory", "persistent_workers",
                        "prefetch_factor", "drop_last", "timeout"}
    for k in _DATALOADER_KEYS:
        shared_dataset_kwargs.pop(k, None)

    # Two-pass construction so that SSP370 (2015-2100) can inherit the
    # historical 1850-1900 climatology baseline for anomaly computation.
    #
    # Pass 1: build all datasets except those that borrow a climatology.
    # Pass 2: wire the historical climatology into SSP370 via
    #         external_climatology, then build those datasets.
    #
    # A scenario "borrows" the hist climatology when:
    #   - its name contains "ssp" (case-insensitive), AND
    #   - it does not set external_climatology itself in its config.
    SSP_BORROWERS = {"ssp370", "ssp245", "ssp585", "ssp126"}

    raw_configs: list[tuple[str, dict]] = []
    for cfg in experiment_configs:
        cfg = dict(cfg)
        name = cfg.pop("scenario_name", None) or f"exp_{len(raw_configs)}"
        raw_configs.append((name, cfg))

    # Pass 1: build non-SSP datasets first to get the hist climatology
    datasets: list[ClimateDataset] = [None] * len(raw_configs)
    scenario_names: list[str]      = [None] * len(raw_configs)
    hist_climatology = None

    # all_years=True swaps in EvalClimateDataset, which loads EVERY year instead
    # of the training subsample (every 5th hist / every other future year). Only
    # for evaluation and generation — training must keep the subsampled default,
    # since that is what every checkpoint was fitted on.
    _DatasetCls = EvalClimateDataset if all_years else ClimateDataset
    if all_years:
        print("[BUILD] all_years=True -> EvalClimateDataset (every year loaded)")

    for i, (name, cfg) in enumerate(raw_configs):
        is_ssp = any(s in name.lower() for s in SSP_BORROWERS)
        has_own_clim = "external_climatology" in cfg
        if is_ssp and not has_own_clim:
            continue  # defer to pass 2
        merged = {**shared_dataset_kwargs, **cfg}
        ds = _DatasetCls(**merged)
        datasets[i] = ds
        scenario_names[i] = name
        # Capture historical climatology (first dataset whose name has "hist")
        if hist_climatology is None and "hist" in name.lower():
            hist_climatology = ds.climatology
            #if hist_climatology is not None:
            #    print(f"[BUILD] Historical climatology captured from '{name}' "
            #          f"shape={tuple(hist_climatology.shape)}")
            #else:
            #    print(f"[BUILD] WARNING: '{name}' has no climatology — "
            #          f"SSP scenarios will fall back to per-batch mean.")

    # Pass 2: build SSP datasets, injecting the hist climatology
    for i, (name, cfg) in enumerate(raw_configs):
        if datasets[i] is not None:
            continue  # already built in pass 1
        merged = {**shared_dataset_kwargs, **cfg}
        if hist_climatology is not None:
            merged["external_climatology"] = hist_climatology
            #print(f"[BUILD] Scenario '{name}': injecting historical climatology "
            #      f"as external_climatology.")
        ds = _DatasetCls(**merged)
        datasets[i] = ds
        scenario_names[i] = name

    multi_ds = MultiExperimentDataset(datasets, scenario_names=scenario_names)
    return MultiExperimentDataLoader(
        multi_ds,
        accelerator,
        batch_size=batch_size,
        mix_scenarios=mix_scenarios,
        stratified=stratified,
        period_boundaries=period_boundaries,
        steps_per_realization=steps_per_realization,
        scenario_weights=scenario_weights,
        prefetch_batches=prefetch_batches,
        year_bias=year_bias,
        year_bias_floor=year_bias_floor,
        bsp_depth=bsp_depth,
        shard_across_ranks=shard_across_ranks,
    )