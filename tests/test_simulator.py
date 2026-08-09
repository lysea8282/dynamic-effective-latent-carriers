import numpy as np

from dynamic_effective_carriers.simulator import simulate_from


def test_two_object_collision_conserves_speed_exchange() -> None:
    initial = np.asarray([[-0.5, 0.0, 1.0, 0.0], [0.5, 0.0, -1.0, 0.0]])
    states = simulate_from(initial, np.zeros((2, 4)))
    assert states.shape == (3, 2, 4)
    assert np.isfinite(states).all()
    assert np.allclose(np.sort(states[-1, :, 2]), [-1.0, 1.0])
