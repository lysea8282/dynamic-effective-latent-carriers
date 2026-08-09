from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CARRIER_DIM = 192


def _canonicalize_column_signs(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class CarrierFit:
    basis: np.ndarray
    singular_values: np.ndarray
    explained_second_moment_ratio: np.ndarray
    unit_count: int
    max_rank: int


def fit_local_edit_delta_pca(deltas: np.ndarray, *, max_rank: int = 64) -> CarrierFit:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != CARRIER_DIM:
        raise ValueError(f"expected an N by {CARRIER_DIM} delta matrix, got {values.shape}")
    if len(values) < max_rank or not np.isfinite(values).all():
        raise ValueError("insufficient or non-finite carrier fit values")
    _, singular_values, right_transpose = np.linalg.svd(values, full_matrices=False)
    basis64 = _canonicalize_column_signs(right_transpose[:max_rank].T)
    error = float(np.max(np.abs(basis64.T @ basis64 - np.eye(max_rank))))
    if error > 1e-10:
        raise RuntimeError(f"basis orthonormality error: {error}")
    second_moment = singular_values**2
    ratios = second_moment / float(np.sum(second_moment))
    return CarrierFit(
        basis64.astype(np.float32),
        singular_values[:max_rank].astype(np.float64),
        ratios[:max_rank].astype(np.float64),
        int(len(values)),
        int(max_rank),
    )


def oracle_coefficients(deltas: np.ndarray, basis: np.ndarray, rank: int) -> np.ndarray:
    return np.asarray(deltas, dtype=np.float32) @ np.asarray(basis, dtype=np.float32)[:, :rank]


def reconstruct_delta(coefficients: np.ndarray, basis: np.ndarray, rank: int, *, beta: float = 1.0) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=np.float32)[..., :rank]
    directions = np.asarray(basis, dtype=np.float32)[:, :rank]
    return (float(beta) * (coeff @ directions.T)).astype(np.float32, copy=False)


def random_orthogonal_equal_norm(reference_delta: np.ndarray, basis: np.ndarray, rank: int, *, seed: int) -> np.ndarray:
    reference = np.asarray(reference_delta, dtype=np.float64)
    directions, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64)[:, :rank], mode="reduced")
    random = np.random.default_rng(int(seed)).standard_normal(reference.shape)
    random -= (random @ directions) @ directions.T
    random_norm = np.linalg.norm(random, axis=1)
    if np.any(random_norm <= 1e-12):
        raise RuntimeError("degenerate random control")
    return (random * (np.linalg.norm(reference, axis=1) / random_norm)[:, None]).astype(np.float32)
