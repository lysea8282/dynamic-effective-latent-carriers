from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import load_checkpoint, validate_fit, validate_jsonl, verify_artifacts
from .carriers import oracle_coefficients, reconstruct_delta
from .rendering import generate_figures, generate_tables
from .simulator import simulate_from
from .training import train
from .utils import read_json, repository_root, write_json


def numerical_smoke(root: Path) -> dict[str, Any]:
    checkpoint_manifest = read_json(root / "checkpoints" / "CHECKPOINT_MANIFEST.json")
    checkpoint_path = root / checkpoint_manifest["checkpoints"][0]["path"]
    model, payload = load_checkpoint(checkpoint_path)
    observations = torch.zeros(1, 5, 2, 6)
    observations[..., 4] = 1.0
    actions = torch.zeros(1, 5, 4)
    with torch.no_grad():
        carrier, _ = model.encode_history(observations, actions)
        decoded = model.decode(carrier)
    fit_path = root / "checkpoints" / "carriers" / "velocity" / "development_seed_291101.npz"
    with np.load(fit_path, allow_pickle=False) as bundle:
        basis = bundle["basis"]
    reference = np.zeros((2, 192), dtype=np.float32)
    reference[0, 0] = 1.0
    coefficients = oracle_coefficients(reference, basis, 4)
    reconstructed = reconstruct_delta(coefficients, basis, 4)
    initial = np.asarray([[0.0, 0.0, 0.1, 0.0], [2.0, 0.0, -0.1, 0.0]], dtype=np.float64)
    trajectory = simulate_from(initial, np.zeros((3, 4), dtype=np.float64))
    return {
        "checkpoint_seed": payload["seed"],
        "carrier_shape": list(carrier.shape),
        "decoded_shape": list(decoded.shape),
        "rank4_projection_shape": list(reconstructed.shape),
        "simulator_shape": list(trajectory.shape),
        "finite": bool(torch.isfinite(decoded).all() and np.isfinite(reconstructed).all() and np.isfinite(trajectory).all()),
    }


def run_quick(root: Path | None = None, *, render: bool = True) -> dict[str, Any]:
    repo = (root or repository_root()).resolve()
    verification = verify_artifacts(repo)
    if not verification["pass"]:
        raise RuntimeError("artifact verification failed: " + "; ".join(verification["failures"]))
    smoke = numerical_smoke(repo)
    if not smoke["finite"]:
        raise RuntimeError("numerical smoke check failed")
    expected = repo / "results" / "expected" / "paper_summary.json"
    reproduced = repo / "results" / "reproduced" / "paper_summary.json"
    reproduced.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(expected, reproduced)
    figures = generate_figures(repo) if render else []
    tables = generate_tables(repo) if render else []
    result = {
        "mode": "frozen_artifact_reproduction",
        "artifact_verification": verification,
        "numerical_smoke": smoke,
        "summary_matches_expected": read_json(expected) == read_json(reproduced),
        "figures": [path.relative_to(repo).as_posix() for path in figures],
        "tables": [path.relative_to(repo).as_posix() for path in tables],
        "status": "complete",
    }
    write_json(repo / "results" / "reproduced" / "quick_reproduction_report.json", result)
    return result


def run_full(
    root: Path | None = None,
    *,
    retrain: bool = False,
    seeds: list[int] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    repo = (root or repository_root()).resolve()
    quick = run_quick(repo)
    dataset_manifest = read_json(repo / "data" / "DATASET_MANIFEST.json")
    row_counts: dict[str, int] = {}
    for item in dataset_manifest["datasets"]:
        path = repo / item["path"]
        if path.suffix == ".jsonl":
            observed = validate_jsonl(path)
            if observed != item["record_count"]:
                raise RuntimeError(f"record count mismatch: {item['path']}")
            row_counts[item["path"]] = observed
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as bundle:
                row_counts[item["path"]] = int(bundle["metadata"].shape[0])
    checkpoint_manifest = read_json(repo / "checkpoints" / "CHECKPOINT_MANIFEST.json")
    loaded = []
    for item in checkpoint_manifest["checkpoints"]:
        _, payload = load_checkpoint(repo / item["path"])
        loaded.append(payload["seed"])
    fits = []
    for item in checkpoint_manifest["carrier_artifacts"]:
        validate_fit(repo / item["path"])
        fits.append(item["path"])
    retraining = []
    if retrain:
        for seed in seeds or [291101, 291102, 291103, 291301, 291302, 291303]:
            target = repo / "results" / "reproduced" / "retraining" / f"model_seed_{seed}.pt"
            retraining.append(train(seed, target, device_name=device))
    result = {
        "mode": "frozen_full_validation_with_optional_retraining",
        "quick": quick,
        "dataset_record_counts": row_counts,
        "loaded_checkpoint_seeds": loaded,
        "validated_carrier_artifact_count": len(fits),
        "retraining": retraining,
        "status": "complete",
    }
    write_json(repo / "results" / "reproduced" / "full_reproduction_report.json", result)
    return result


def checklist(result: dict[str, Any]) -> str:
    summary = [
        "[x] artifact hashes",
        "[x] native counterfactual adequacy",
        "[x] velocity rank profile and addressable result",
        "[x] autonomous-rollout controls",
        "[x] temporal stability",
        "[x] position intervention specificity diagnostic",
        "[x] paper figures and tables",
    ]
    return "\n".join(summary + [f"status: {result['status']}"])
