from __future__ import annotations

import numpy as np


def apply_state_edit(state: np.ndarray, object_id: int, coordinate: int, delta: float) -> np.ndarray:
    edited = np.asarray(state, dtype=np.float64).copy()
    if edited.shape != (2, 4) or object_id not in (0, 1) or coordinate not in range(4):
        raise ValueError("expected state shape (2, 4), object 0 or 1, and coordinate 0 through 3")
    edited[object_id, coordinate] += float(delta)
    return edited
