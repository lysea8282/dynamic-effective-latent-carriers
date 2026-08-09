from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import DeterministicBeliefModel, autonomous_sequence, parameter_count
from .simulator.two_ball import NORMALIZER_MEAN, NORMALIZER_STD, normalize_states
from .utils import repository_root, sha256, write_json

HORIZON = 12
EPOCHS = 70
BATCHES_PER_EPOCH = 24
TOTAL_UPDATES = EPOCHS * BATCHES_PER_EPOCH
UNITS_PER_BATCH = 96


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest().upper()


def transition_weights(physical: torch.Tensor, q90: float = 1.276299218698019) -> torch.Tensor:
    delta = physical[:, 1:, :, 2:] - physical[:, :-1, :, 2:]
    magnitude = torch.sqrt(torch.sum(delta * delta, dim=(2, 3)))
    return torch.where(magnitude > 1e-8, 1.0 + torch.clamp(magnitude / q90, max=1.0), torch.ones_like(magnitude))


def model_loss(
    model: DeterministicBeliefModel,
    observations: torch.Tensor,
    actions: torch.Tensor,
    truth: torch.Tensor,
    physical: torch.Tensor,
    anchor: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    carrier, info = model.encode_history(observations[:, : anchor + 1], actions[:, : anchor + 1])
    sequence = autonomous_sequence(model, carrier, actions, anchor, HORIZON)
    target = truth[:, anchor : anchor + HORIZON + 1]
    rollout = torch.mean((sequence - target) ** 2)
    anchor_loss = torch.mean((sequence[:, 0] - target[:, 0]) ** 2)
    filtered = torch.stack([model.decode(value).reshape(-1, 2, 4) for value in info["saved_carriers"]], dim=1)
    filtering = torch.mean((filtered - truth[:, : anchor + 1]) ** 2)
    pred_r = sequence[:, :, 1, :2] - sequence[:, :, 0, :2]
    true_r = target[:, :, 1, :2] - target[:, :, 0, :2]
    pred_u = sequence[:, :, 1, 2:] - sequence[:, :, 0, 2:]
    true_u = target[:, :, 1, 2:] - target[:, :, 0, 2:]
    relative = torch.mean(torch.mean((pred_r - true_r) ** 2, dim=-1) + torch.mean((pred_u - true_u) ** 2, dim=-1))
    weights = transition_weights(physical[:, anchor : anchor + HORIZON + 1])
    pred_dv = sequence[:, 1:, :, 2:] - sequence[:, :-1, :, 2:]
    true_dv = target[:, 1:, :, 2:] - target[:, :-1, :, 2:]
    delta_v = torch.sum(weights * torch.mean((pred_dv - true_dv) ** 2, dim=(2, 3))) / torch.sum(weights)
    total = rollout + 0.20 * anchor_loss + 0.20 * filtering + 0.25 * relative + 0.25 * delta_v
    return total, {"rollout": rollout, "anchor": anchor_loss, "filter": filtering, "relative": relative, "delta_v": delta_v}


def _load_dataset(path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files if name != "metadata"}
        metadata = [json.loads(str(value)) for value in bundle["metadata"]]
    return arrays, metadata


def train(
    seed: int,
    output: Path,
    *,
    updates: int = TOTAL_UPDATES,
    units_per_batch: int = UNITS_PER_BATCH,
    device_name: str | None = None,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    source = dataset_path or repository_root() / "data" / "training" / "model_training_dataset.npz"
    arrays, records = _load_dataset(source)
    train_indices = np.asarray([i for i, row in enumerate(records) if row["split"] == "MODEL_TRAIN"], dtype=np.int64)
    anchors = np.asarray([row["anchor"] for row in records], dtype=np.int64)
    by_anchor = {anchor: train_indices[anchors[train_indices] == anchor] for anchor in range(5, 10)}
    model = DeterministicBeliefModel().to(device)
    if parameter_count(model) != 496200:
        raise RuntimeError("parameter-count mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=0.00015)
    rng = np.random.default_rng(seed + 17)
    to_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    device_arrays = {
        "actions": to_tensor(arrays["actions"]),
        "factual_obs": to_tensor(arrays["factual_obs"]),
        "counterfactual_obs": to_tensor(arrays["counterfactual_obs"]),
        "factual_truth": to_tensor(normalize_states(arrays["factual"])),
        "counterfactual_truth": to_tensor(normalize_states(arrays["counterfactual"])),
        "factual_physical": to_tensor(arrays["factual"]),
        "counterfactual_physical": to_tensor(arrays["counterfactual"]),
    }
    traces: list[dict[str, float]] = []
    started = time.time()
    model.train()
    for update in range(updates):
        anchor = 5 + update % 5
        population = by_anchor[anchor]
        indices = rng.choice(population, size=units_per_batch, replace=len(population) < units_per_batch)
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        actions = device_arrays["actions"].index_select(0, index)
        combined = (
            torch.cat((device_arrays["factual_obs"].index_select(0, index), device_arrays["counterfactual_obs"].index_select(0, index))),
            torch.cat((actions, actions)),
            torch.cat((device_arrays["factual_truth"].index_select(0, index), device_arrays["counterfactual_truth"].index_select(0, index))),
            torch.cat((device_arrays["factual_physical"].index_select(0, index), device_arrays["counterfactual_physical"].index_select(0, index))),
        )
        optimizer.zero_grad(set_to_none=True)
        loss, terms = model_loss(model, *combined, anchor)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {update + 1}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
        optimizer.step()
        if (update + 1) % BATCHES_PER_EPOCH == 0:
            scheduler.step()
        if update < 3 or (update + 1) % 120 == 0 or update + 1 == updates:
            traces.append({
                "update": update + 1,
                "anchor": anchor,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm,
                "total": float(loss.detach().cpu()),
                **{name: float(value.detach().cpu()) for name, value in terms.items()},
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "public-model-checkpoint-v1",
        "model_id": f"model_seed_{seed}",
        "seed": seed,
        "label": "retrained_replication",
        "objective_id": "equal_factual_local_counterfactual_v2",
        "optimizer_updates": updates,
        "parameter_count": parameter_count(model),
        "carrier_dimension": 192,
        "normalizer_mean": NORMALIZER_MEAN.tolist(),
        "normalizer_std": NORMALIZER_STD.tolist(),
        "state_dict": model.state_dict(),
        "state_hash": state_hash(model),
        "loss_trace": traces,
    }
    torch.save(payload, output)
    try:
        reported_path = output.relative_to(repository_root()).as_posix()
    except ValueError:
        reported_path = output.name
    result = {
        "seed": seed,
        "checkpoint_path": reported_path,
        "checkpoint_size_bytes": output.stat().st_size,
        "checkpoint_sha256": sha256(output),
        "state_hash": payload["state_hash"],
        "optimizer_updates": updates,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "status": "complete",
        "loss_trace": traces,
    }
    write_json(output.with_suffix(".json"), result)
    return result
