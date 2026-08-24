from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from reproduce_corrected_joint_results import reproduce
from verify_joint_alignment import EXPECTED_MEMBERSHIP_SHA256, verify_alignment


def test_joint_source_identity_and_numeric_semantics() -> None:
    result = verify_alignment()
    assert result["status"] == "PASS", result["failures"]
    assert result["identity_join"] == "public_unit_id"
    assert result["membership_sha256"] == EXPECTED_MEMBERSHIP_SHA256
    assert result["canonical_unit_count"] == 512
    assert result["fit_unit_count"] == 411
    assert result["test_unit_count"] == 101


def test_corrected_joint_public_summaries_recompute() -> None:
    result = reproduce()
    assert result["status"] == "PASS", result["failures"]
    assert result["cardinalities"]["addressability_unit_rows"] == 606
    assert result["cardinalities"]["specificity_control_rows"] == 2424
