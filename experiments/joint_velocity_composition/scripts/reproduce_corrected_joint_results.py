from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_joint_alignment import verify_alignment


TOLERANCE = 1e-6


def _rows(name: str, package: Path = PACKAGE) -> list[dict[str, str]]:
    with (package / "data" / name).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _median(values: Iterable[str | float]) -> float:
    return float(statistics.median(float(value) for value in values))


def _assert_close(failures: list[str], label: str, observed: float, expected: float) -> None:
    if abs(observed - expected) > TOLERANCE:
        failures.append(f"{label}: observed={observed} expected={expected}")


def reproduce(package: Path = PACKAGE) -> dict[str, Any]:
    failures: list[str] = []
    alignment = verify_alignment(package)
    if alignment["status"] != "PASS":
        failures.extend(f"alignment: {item}" for item in alignment["failures"])

    behavior = _rows("joint_behavior_summary.csv", package)
    behavior_checkpoint = [row for row in behavior if row["scope"] == "checkpoint"]
    behavior_family = {row["family"]: row for row in behavior if row["scope"] == "family_median"}
    for family in ("M0", "M1", "M2"):
        group = [row for row in behavior_checkpoint if row["family"] == family]
        for name in ("coverage", "anchor_rmse", "k1_rmse", "t19_rmse"):
            _assert_close(
                failures,
                f"behavior/{family}/{name}",
                _median(row[name] for row in group),
                float(behavior_family[family][name]),
            )

    composition_units = _rows("joint_composition_units.csv", package)
    composition = _rows("joint_composition_summary.csv", package)
    composition_checkpoint = {
        (row["family"], row["seed"]): row for row in composition if row["scope"] == "checkpoint"
    }
    unit_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in composition_units:
        unit_groups[(row["family"], row["seed"])].append(row)
    for key, group in unit_groups.items():
        if len(group) != 101 or len({row["public_unit_id"] for row in group}) != 101:
            failures.append(f"composition/{key}: expected 101 unique units")
        expected_row = composition_checkpoint[key]
        for name in ("E_add", "R_norm", "R_norm_error", "cos_add", "cos_xy"):
            _assert_close(
                failures,
                f"composition/{key}/{name}",
                _median(row[name] for row in group),
                float(expected_row[name]),
            )

    dynamic = _rows("joint_dynamic_composition.csv", package)
    addressability = _rows("joint_addressability_summary.csv", package)
    addressability_expected = {(row["family"], row["seed"]): row for row in addressability}
    addressability_units = _rows("joint_addressability_units.csv", package)
    address_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    intended_by_unit: dict[tuple[str, str, str], float] = {}
    for row in addressability_units:
        key = (row["family"], row["seed"])
        address_groups[key].append(row)
        intended_by_unit[(row["family"], row["seed"], row["public_unit_id"])] = float(row["M1"])
    for key, group in address_groups.items():
        if len(group) != 101 or len({row["public_unit_id"] for row in group}) != 101:
            failures.append(f"addressability/{key}: expected 101 unique units")
        expected_row = addressability_expected[key]
        observed = {
            "Addressable_coverage": sum(row["joint_pass"] == "true" for row in group) / len(group),
            "Addressable_anchor_RMSE": _median(row["anchor_rmse"] for row in group),
            "Addressable_k1_RMSE": _median(row["k1_rmse"] for row in group),
            "Addressable_t19_RMSE": _median(row["t19_rmse"] for row in group),
        }
        for name, value in observed.items():
            _assert_close(failures, f"addressability/{key}/{name}", value, float(expected_row[name]))

    control_units = _rows("joint_specificity_controls.csv", package)
    control_summary = _rows("joint_specificity_summary.csv", package)
    control_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in control_units:
        control_groups[(row["family"], row["seed"], row["control_type"])].append(row)
    control_expected = {
        (row["family"], row["seed"], row["control_type"]): row for row in control_summary
    }
    for key, group in control_groups.items():
        if len(group) != 101 or len({row["public_unit_id"] for row in group}) != 101:
            failures.append(f"specificity/{key}: expected 101 unique units")
        intended = [intended_by_unit[(key[0], key[1], row["public_unit_id"])] for row in group]
        controls = [float(row["M1"]) for row in group]
        expected_row = control_expected[key]
        observed_intended = _median(intended)
        observed_control = _median(controls)
        observed_fraction = sum(control > target for control, target in zip(controls, intended)) / len(group)
        for name, value in (
            ("intended_M1_median", observed_intended),
            ("control_M1_median", observed_control),
            ("separation", observed_control - observed_intended),
            ("fraction_control_worse_than_intended", observed_fraction),
        ):
            _assert_close(failures, f"specificity/{key}/{name}", value, float(expected_row[name]))

    geometry_units = _rows("joint_patch_geometry_units.csv", package)
    geometry_summary = {
        (row["family"], row["seed"]): row for row in _rows("joint_patch_geometry.csv", package)
    }
    geometry_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in geometry_units:
        geometry_groups[(row["family"], row["seed"])].append(row)
    for key, group in geometry_groups.items():
        if len(group) != 101 or len({row["public_unit_id"] for row in group}) != 101:
            failures.append(f"patch_geometry/{key}: expected 101 unique units")
        for name in ("E_full_oracle", "E_request_oracle", "E_base_oracle"):
            _assert_close(
                failures,
                f"patch_geometry/{key}/{name}",
                _median(row[name] for row in group),
                float(geometry_summary[key][name]),
            )

    subspace = _rows("joint_subspace_composition.csv", package)
    for family in ("M1", "M2"):
        checkpoints = [row for row in subspace if row["scope"] == "checkpoint" and row["family"] == family]
        family_row = next(
            row for row in subspace if row["scope"] == "family_median" and row["family"] == family
        )
        for name in (
            "E_add_total",
            "E_add_U4",
            "E_add_comp",
            "defect_energy_U4_fraction",
            "defect_energy_comp_fraction",
        ):
            _assert_close(
                failures,
                f"subspace/{family}/{name}",
                _median(row[name] for row in checkpoints),
                float(family_row[name]),
            )

    expected = json.loads((package / "expected" / "corrected_joint_summary.json").read_text(encoding="utf-8"))
    for family in ("M0", "M1", "M2"):
        expected_family = expected["family_medians"][family]
        for name in ("joint_coverage", "anchor_rmse", "k1_rmse", "t19_rmse"):
            public_name = "coverage" if name == "joint_coverage" else name
            _assert_close(
                failures,
                f"expected/{family}/{name}",
                float(behavior_family[family][public_name]),
                float(expected_family[name]),
            )
        static_family = next(
            row for row in composition if row["scope"] == "family_median" and row["family"] == family
        )
        _assert_close(failures, f"expected/{family}/E_add", float(static_family["E_add"]), expected_family["E_add"])
        _assert_close(
            failures,
            f"expected/{family}/E_dyn12",
            _median(row["E_dyn"] for row in dynamic if row["family"] == family and row["k"] == "12"),
            expected_family["E_dyn12"],
        )

    cardinalities = {
        "canonical_units": alignment["canonical_unit_count"],
        "composition_unit_rows": len(composition_units),
        "addressability_unit_rows": len(addressability_units),
        "specificity_control_rows": len(control_units),
        "patch_geometry_unit_rows": len(geometry_units),
        "checkpoint_behavior_rows": len(behavior_checkpoint),
        "checkpoint_addressability_rows": len(addressability),
    }
    expected_cardinalities = {
        "canonical_units": 512,
        "composition_unit_rows": 909,
        "addressability_unit_rows": 606,
        "specificity_control_rows": 2424,
        "patch_geometry_unit_rows": 606,
        "checkpoint_behavior_rows": 9,
        "checkpoint_addressability_rows": 6,
    }
    if cardinalities != expected_cardinalities:
        failures.append(f"cardinality mismatch: {cardinalities}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "numeric_tolerance": TOLERANCE,
        "alignment": alignment,
        "cardinalities": cardinalities,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the corrected Joint public results.")
    parser.add_argument(
        "--figure",
        action="store_true",
        help="also render the corrected three-panel public draft figure",
    )
    args = parser.parse_args()
    result = reproduce()
    if args.figure and result["status"] == "PASS":
        repository = PACKAGE.parents[1]
        figure_script = repository / "figures" / "joint_extension" / "make_joint_extension_figure.py"
        spec = importlib.util.spec_from_file_location("joint_extension_figure", figure_script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load figure script: {figure_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outputs = module.render(figure_script.parent / "figure_joint_extension")
        result["figure_outputs"] = [path.relative_to(repository).as_posix() for path in outputs]
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
