from pathlib import Path

import torch

from dynamic_effective_carriers.artifacts import load_checkpoint


def test_model_checkpoint_load_and_forward() -> None:
    root = Path(__file__).resolve().parents[1]
    model, payload = load_checkpoint(root / "checkpoints/models/development/model_seed_291101.pt")
    observations = torch.zeros(2, 5, 2, 6)
    actions = torch.zeros(2, 5, 4)
    with torch.no_grad():
        carrier, _ = model.encode_history(observations, actions)
        decoded = model.decode(carrier)
    assert payload["parameter_count"] == 496200
    assert carrier.shape == (2, 192)
    assert decoded.shape == (2, 8)
