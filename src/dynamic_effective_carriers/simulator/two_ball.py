from __future__ import annotations

import math

import numpy as np

DT = 0.25
RADIUS = 0.25
ARENA = 4.75
NORMALIZER_MEAN = np.asarray(
    [0.010910557582974434, 0.008827612735331059, -0.0013239462859928608, 0.00020721465989481658] * 2,
    dtype=np.float32,
)
NORMALIZER_STD = np.asarray(
    [0.6748629808425903, 0.6790841221809387, 0.2936058044433594, 0.2959529161453247] * 2,
    dtype=np.float32,
)


def time_of_impact(positions: np.ndarray, velocities: np.ndarray) -> float | None:
    relative_position = positions[1] - positions[0]
    relative_velocity = velocities[1] - velocities[0]
    if float(relative_position @ relative_velocity) >= 0:
        return None
    a = float(relative_velocity @ relative_velocity)
    b = 2.0 * float(relative_position @ relative_velocity)
    c = float(relative_position @ relative_position) - (2.0 * RADIUS) ** 2
    discriminant = b * b - 4.0 * a * c
    if a <= 1e-12 or discriminant < -1e-10:
        return None
    hit = (-b - math.sqrt(max(discriminant, 0.0))) / (2.0 * a)
    return float(np.clip(hit, 0.0, DT)) if -1e-10 <= hit <= DT + 1e-10 else None


def simulate_from(initial: np.ndarray, actions: np.ndarray) -> np.ndarray:
    states = np.zeros((len(actions) + 1, 2, 4), dtype=np.float64)
    states[0] = np.asarray(initial, dtype=np.float64)
    for step, action in enumerate(np.asarray(actions, dtype=np.float64)):
        positions = states[step, :, :2].copy()
        velocities = states[step, :, 2:].copy()
        if np.any(action[:2]):
            velocities[int(np.argmax(action[:2]))] += action[2:]
        hit = time_of_impact(positions, velocities)
        if hit is None:
            positions += velocities * DT
        else:
            positions += velocities * hit
            displacement = positions[1] - positions[0]
            normal = displacement / max(float(np.linalg.norm(displacement)), 1e-12)
            impulse = -float((velocities[1] - velocities[0]) @ normal)
            velocities[0] -= impulse * normal
            velocities[1] += impulse * normal
            positions += velocities * (DT - hit)
        for slot in range(2):
            for axis in range(2):
                while positions[slot, axis] > ARENA or positions[slot, axis] < -ARENA:
                    if positions[slot, axis] > ARENA:
                        positions[slot, axis] = 2.0 * ARENA - positions[slot, axis]
                        velocities[slot, axis] *= -1.0
                    if positions[slot, axis] < -ARENA:
                        positions[slot, axis] = -2.0 * ARENA - positions[slot, axis]
                        velocities[slot, axis] *= -1.0
        states[step + 1, :, :2] = positions
        states[step + 1, :, 2:] = velocities
    return states


def normalize_states(states: np.ndarray) -> np.ndarray:
    shape = states.shape
    flat = np.asarray(states, dtype=np.float32).reshape(*shape[:-2], 8)
    return ((flat - NORMALIZER_MEAN) / NORMALIZER_STD).reshape(shape).astype(np.float32)


def denormalize_states(states: np.ndarray) -> np.ndarray:
    shape = states.shape
    flat = np.asarray(states, dtype=np.float32).reshape(*shape[:-2], 8)
    return (flat * NORMALIZER_STD + NORMALIZER_MEAN).reshape(shape).astype(np.float32)


def build_observations(states: np.ndarray, noise: np.ndarray, anchor: int) -> np.ndarray:
    observations = np.zeros((len(states), 2, 6), dtype=np.float32)
    visible = np.zeros(len(states), dtype=np.float32)
    visible[: min(5, anchor)] = 1.0
    visible[anchor] = 1.0
    observations[..., :4] = np.asarray(states, dtype=np.float32) + np.asarray(noise, dtype=np.float32)
    observations[..., 4] = visible[:, None]
    observations[:, 0, 5] = -1.0
    observations[:, 1, 5] = 1.0
    for slot in range(2):
        lo = slot * 4
        observations[:, slot, :4] = (
            (observations[:, slot, :4] - NORMALIZER_MEAN[lo : lo + 4]) / NORMALIZER_STD[lo : lo + 4]
        ) * visible[:, None]
    return observations


def contact_steps(states: np.ndarray) -> list[int]:
    distances = np.linalg.norm(states[:, 1, :2] - states[:, 0, :2], axis=-1)
    velocity_jump = np.linalg.norm(np.diff(states[:, :, 2:], axis=0), axis=(1, 2))
    return [int(i) for i in range(1, len(states)) if distances[i] <= 0.500001 or velocity_jump[i - 1] > 1e-5]
