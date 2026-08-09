from pathlib import Path

from dynamic_effective_carriers.artifacts import verify_artifacts


def test_packaged_artifact_hashes() -> None:
    result = verify_artifacts(Path(__file__).resolve().parents[1])
    assert result["pass"], result["failures"]
    assert result["checked"] >= 100
