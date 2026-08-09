from __future__ import annotations

import numpy as np


def normalized_effect(factual: np.ndarray, counterfactual: np.ndarray, floor: float = 1e-8) -> float:
    factual_array = np.asarray(factual, dtype=np.float64)
    delta = np.asarray(counterfactual, dtype=np.float64) - factual_array
    return float(np.linalg.norm(delta) / max(float(np.linalg.norm(factual_array)), floor))
