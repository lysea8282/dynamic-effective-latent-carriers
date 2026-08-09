from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import DeterministicBeliefModel, parameter_count
from .utils import read_json, repository_root, sha256


def verify_artifacts(root: Path | None = None) -> dict[str, Any]:
    repo = (root or repository_root()).resolve()
    manifest = read_json(repo / "ARTIFACT_MANIFEST.json")
    failures: list[str] = []
    checked = 0
    for item in manifest["artifacts"]:
        relative = Path(item["path"])
        path = (repo / relative).resolve()
        if repo not in path.parents:
            failures.append(f"path escapes repository: {relative.as_posix()}")
            continue
        if not path.is_file():
            failures.append(f"missing: {relative.as_posix()}")
            continue
        checked += 1
        if path.stat().st_size != item["size_bytes"]:
            failures.append(f"size mismatch: {relative.as_posix()}")
        if sha256(path) != item["sha256"]:
            failures.append(f"hash mismatch: {relative.as_posix()}")
    return {"pass": not failures, "checked": checked, "failures": failures}


def load_checkpoint(path: Path) -> tuple[DeterministicBeliefModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DeterministicBeliefModel()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if parameter_count(model) != 496200 or payload["carrier_dimension"] != 192:
        raise RuntimeError("checkpoint topology mismatch")
    return model, payload


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def validate_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not _finite_numbers(row):
                raise ValueError(f"non-finite number in {path.name} line {line_number}")
            count += 1
    return count


def validate_fit(path: Path) -> None:
    with np.load(path, allow_pickle=False) as bundle:
        for name in bundle.files:
            values = bundle[name]
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                raise ValueError(f"non-finite array {name} in {path.name}")
        if "basis" in bundle.files:
            basis = np.asarray(bundle["basis"], dtype=np.float64)
            error = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))))
            if error > 2e-5:
                raise ValueError(f"non-orthonormal basis in {path.name}: {error}")
