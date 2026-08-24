from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parents[1]
EXPECTED_MEMBERSHIP_SHA256 = "cec12621b562bfdb7c09e381e6fd17eaa6bb16435ece64060d07f33ec496bd9f"
NUMERIC_TOLERANCE = 2e-7


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_alignment(package: Path = PACKAGE) -> dict[str, Any]:
    rows = _rows(package / "data" / "joint_request_units.csv")
    failures: list[str] = []
    identifiers = [row["public_unit_id"] for row in rows]
    indices = [int(row["source_record_index"]) for row in rows]
    membership_sha256 = hashlib.sha256("\n".join(sorted(identifiers)).encode("utf-8")).hexdigest()

    if len(rows) != 512:
        failures.append(f"expected 512 canonical units, found {len(rows)}")
    if len(set(identifiers)) != len(identifiers):
        failures.append("duplicate public unit identity")
    if sorted(indices) != list(range(512)):
        failures.append("missing or duplicate source record index")
    if membership_sha256 != EXPECTED_MEMBERSHIP_SHA256:
        failures.append("canonical public unit membership digest mismatch")

    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("FIT", "TEST")
    }
    if split_counts != {"FIT": 411, "TEST": 101}:
        failures.append(f"unexpected split counts: {split_counts}")

    # All semantic comparisons are performed after an explicit identity-keyed
    # reconstruction. Row order is deliberately reversed so no positional
    # fixture-to-record binding can satisfy this check accidentally.
    by_identity = {row["public_unit_id"]: row for row in reversed(rows)}
    max_metadata_numeric_error = 0.0
    max_composition_error = 0.0
    for identifier in sorted(identifiers):
        row = by_identity[identifier]
        if row["edited_object"] != row["numeric_edited_object"]:
            failures.append(f"edited-object mismatch for {identifier}")
            continue
        metadata_numeric_error = max(
            abs(float(row["metadata_delta_vx"]) - float(row["numeric_delta_vx"])),
            abs(float(row["metadata_delta_vy"]) - float(row["numeric_delta_vy"])),
        )
        max_metadata_numeric_error = max(max_metadata_numeric_error, metadata_numeric_error)
        composition_error = max(
            abs(float(row["single_x_delta_vx"]) + float(row["single_y_delta_vx"]) - float(row["joint_delta_vx"])),
            abs(float(row["single_x_delta_vy"]) + float(row["single_y_delta_vy"]) - float(row["joint_delta_vy"])),
            float(row["composition_max_abs_error"]),
        )
        max_composition_error = max(max_composition_error, composition_error)
        semantic_flags = (
            "preanchor_observations_identical",
            "anchor_positions_identical",
            "same_source_noise_identity",
            "semantic_pass",
        )
        if any(row[name] != "true" for name in semantic_flags):
            failures.append(f"semantic identity flag failed for {identifier}")

    if max_metadata_numeric_error > NUMERIC_TOLERANCE:
        failures.append(f"metadata/numerical edit error exceeds tolerance: {max_metadata_numeric_error}")
    if max_composition_error > NUMERIC_TOLERANCE:
        failures.append(f"Single-x + Single-y != Joint within tolerance: {max_composition_error}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "identity_join": "public_unit_id",
        "canonical_unit_count": len(rows),
        "fit_unit_count": split_counts["FIT"],
        "test_unit_count": split_counts["TEST"],
        "membership_sha256": membership_sha256,
        "max_metadata_numeric_error": max_metadata_numeric_error,
        "max_single_sum_joint_error": max_composition_error,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "failures": failures,
    }


def main() -> None:
    result = verify_alignment()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
