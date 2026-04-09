"""
Sampler module to handle balanced sampling strategies
and oversampling techniques for imbalanced continuous targets.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedSampler(Sampler[list[int]]):
    """
    Multidimensional balanced batch sampler with optional tail oversampling.

    This sampler yields lists of indices (one list per batch) and is meant to be used
    as a ``batch_sampler`` in a DataLoader.

    Strategy:
    - Build fixed-width bins per target dimension.
    - Compute inverse-frequency rarity per sample.
    - Oversample the rare-tail subset in every batch.
    - Fill remaining batch slots with rotating (dimension, bin) coverage picks.

    Notes:
    - If ``batch_size`` is smaller than ``num_bins * num_target_dims``, full per-batch
      bin coverage is impossible; the sampler rotates coverage across batches.
    - Sampling uses replacement by default to allow stable tail oversampling.
    """

    def __init__(
        self,
        data_source,
        batch_size: int,
        target_attr: str = "y",
        target_dims: list[int] | None = None,
        num_bins: int = 10,
        tail_fraction: float = 0.2,
        tail_ratio_in_batch: float = 0.5,
        replacement: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if not (0.0 <= tail_fraction <= 1.0):
            raise ValueError("tail_fraction must be in [0, 1]")
        if not (0.0 <= tail_ratio_in_batch <= 1.0):
            raise ValueError("tail_ratio_in_batch must be in [0, 1]")

        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.target_attr = target_attr
        self.num_bins = int(num_bins)
        self.tail_fraction = float(tail_fraction)
        self.tail_ratio_in_batch = float(tail_ratio_in_batch)
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self._rng = np.random.default_rng(seed)

        self._targets = self._extract_targets()
        n_samples, n_dims_total = self._targets.shape
        if n_samples == 0:
            raise ValueError("data_source is empty")

        if target_dims is None:
            self.target_dims = list(range(n_dims_total))
        else:
            self.target_dims = [d for d in target_dims if 0 <= d < n_dims_total]
            if not self.target_dims:
                raise ValueError("target_dims contains no valid dimension indices")

        self._bin_ids_per_dim: dict[int, np.ndarray] = {}
        self._counts_per_dim: dict[int, np.ndarray] = {}
        self._indices_by_dim_bin: dict[int, dict[int, np.ndarray]] = {}
        self._bin_cycle_pool: dict[int, dict[int, np.ndarray]] = {}
        self._bin_cycle_ptr: dict[int, dict[int, int]] = {}

        # Mean inverse-frequency rarity across selected dimensions.
        rarity_parts = []
        for dim in self.target_dims:
            values = self._targets[:, dim]
            bin_ids = self._compute_bin_ids(values)
            counts = np.bincount(bin_ids, minlength=self.num_bins).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            mean_count = float(np.mean(counts))
            rarity = mean_count / counts[bin_ids]
            rarity_parts.append(rarity)

            self._bin_ids_per_dim[dim] = bin_ids
            self._counts_per_dim[dim] = counts

            dim_map: dict[int, np.ndarray] = {}
            dim_pool: dict[int, np.ndarray] = {}
            dim_ptr: dict[int, int] = {}
            for b in range(self.num_bins):
                idx = np.where(bin_ids == b)[0]
                dim_map[b] = idx
                if idx.size == 0:
                    dim_pool[b] = np.empty(0, dtype=np.int64)
                    dim_ptr[b] = 0
                else:
                    shuffled = self._rng.permutation(idx)
                    dim_pool[b] = shuffled
                    dim_ptr[b] = 0
            self._indices_by_dim_bin[dim] = dim_map
            self._bin_cycle_pool[dim] = dim_pool
            self._bin_cycle_ptr[dim] = dim_ptr

        self._rarity = np.mean(np.stack(rarity_parts, axis=0), axis=0)
        self._all_indices = np.arange(n_samples, dtype=np.int64)

        if self.tail_fraction > 0.0:
            threshold = float(np.quantile(self._rarity, 1.0 - self.tail_fraction))
            tail_idx = np.where(self._rarity >= threshold)[0]
            self._tail_indices = tail_idx if tail_idx.size > 0 else self._all_indices
        else:
            self._tail_indices = np.empty(0, dtype=np.int64)

        probs = self._rarity.astype(np.float64)
        probs_sum = float(probs.sum())
        self._rarity_probs = probs / probs_sum if probs_sum > 0 else np.full(n_samples, 1.0 / n_samples)

        self._coverage_pairs = [(d, b) for d in self.target_dims for b in range(self.num_bins)]
        self._coverage_cursor = 0

    def _extract_targets(self) -> np.ndarray:
        targets = []
        for data in self.data_source:
            y = getattr(data, self.target_attr)
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
            y = y.detach().cpu().reshape(-1).to(dtype=torch.float32)
            targets.append(y.numpy())

        arr = np.stack(targets, axis=0)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def _compute_bin_ids(self, values: np.ndarray) -> np.ndarray:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros(values.shape[0], dtype=np.int64)

        edges = np.linspace(vmin, vmax, self.num_bins + 1, dtype=np.float64)
        bin_ids = np.digitize(values, edges[1:-1], right=True)
        return np.clip(bin_ids, 0, self.num_bins - 1).astype(np.int64)

    def _draw_from_pool(self, pool: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        if pool.size == 0:
            return np.empty(0, dtype=np.int64)

        if self.replacement:
            return self._rng.choice(pool, size=k, replace=True).astype(np.int64)

        if k <= pool.size:
            return self._rng.choice(pool, size=k, replace=False).astype(np.int64)

        # Fallback: consume all unique once, then top up with replacement.
        unique = self._rng.choice(pool, size=pool.size, replace=False).astype(np.int64)
        extra = self._rng.choice(pool, size=k - pool.size, replace=True).astype(np.int64)
        return np.concatenate([unique, extra], axis=0)

    def _next_from_dim_bin_cycle(self, dim: int, bin_idx: int) -> int | None:
        pool = self._bin_cycle_pool[dim][bin_idx]
        if pool.size == 0:
            return None

        ptr = self._bin_cycle_ptr[dim][bin_idx]
        if ptr >= pool.size:
            pool = self._rng.permutation(pool)
            self._bin_cycle_pool[dim][bin_idx] = pool
            ptr = 0

        idx = int(pool[ptr])
        self._bin_cycle_ptr[dim][bin_idx] = ptr + 1
        return idx

    def _build_one_batch(self) -> list[int]:
        batch = []
        tail_k = int(round(self.batch_size * self.tail_ratio_in_batch))
        tail_k = min(tail_k, self.batch_size)

        if tail_k > 0 and self._tail_indices.size > 0:
            batch.extend(self._draw_from_pool(self._tail_indices, tail_k).tolist())

        remaining = self.batch_size - len(batch)
        if remaining <= 0:
            self._rng.shuffle(batch)
            return batch[: self.batch_size]

        # Rotate over all (dimension, bin) pairs to maximize coverage over batches.
        coverage_take = min(remaining, len(self._coverage_pairs))
        for _ in range(coverage_take):
            dim, b = self._coverage_pairs[self._coverage_cursor]
            self._coverage_cursor = (self._coverage_cursor + 1) % len(self._coverage_pairs)
            idx = self._next_from_dim_bin_cycle(dim, b)
            if idx is not None:
                batch.append(idx)

        remaining = self.batch_size - len(batch)
        if remaining > 0:
            batch.extend(
                self._rng.choice(self._all_indices, size=remaining, replace=True, p=self._rarity_probs)
                .astype(np.int64)
                .tolist()
            )

        self._rng.shuffle(batch)
        return batch

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.data_source)
        n_batches = n // self.batch_size
        if not self.drop_last and n % self.batch_size != 0:
            n_batches += 1

        for _ in range(n_batches):
            yield self._build_one_batch()

    def __len__(self) -> int:
        n = len(self.data_source)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)