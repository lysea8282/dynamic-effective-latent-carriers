from __future__ import annotations

import numpy as np


def equal_norm_random(reference: np.ndarray, seed: int) -> np.ndarray:
    value = np.asarray(reference, dtype=np.float64)
    random = np.random.default_rng(int(seed)).standard_normal(value.shape)
    norm = float(np.linalg.norm(random))
    return (random * (float(np.linalg.norm(value)) / max(norm, 1e-12))).astype(np.float32)
