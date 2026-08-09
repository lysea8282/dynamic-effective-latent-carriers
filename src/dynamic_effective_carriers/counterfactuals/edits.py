from __future__ import annotations

import numpy as np

from ..interventions import apply_state_edit


def paired_state_edit(state: np.ndarray, object_id: int, coordinate: int, delta: float) -> tuple[np.ndarray, np.ndarray]:
    factual = np.asarray(state, dtype=np.float64).copy()
    return factual, apply_state_edit(factual, object_id, coordinate, delta)
