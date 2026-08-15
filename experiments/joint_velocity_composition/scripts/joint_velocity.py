"""Public joint-velocity evidence replay helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = EXPERIMENT_ROOT / "data"
EXPECTED_ROOT = EXPERIMENT_ROOT / "expected"
REPRODUCED_ROOT = PUBLIC_ROOT / "results" / "reproduced" / "joint_velocity_composition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def as_bool(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def tri_bool(value: str) -> bool | None:
    if value.lower() in {"n/a", "na", "none"}:
        return None
    return as_bool(value)


def verify_public_sources() -> dict[str, Any]:
    manifest = read_json(DATA_ROOT / "public_source_manifest.json")
    failures = []
    checked = 0
    for item in manifest["files"]:
        path = DATA_ROOT / item["name"]
        if not path.is_file():
            failures.append(f"missing data file: {item['name']}")
            continue
        checked += 1
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            failures.append(f"data identity mismatch: {item['name']}")
    for item in manifest["code_files"]:
        path = PUBLIC_ROOT / item["path"]
        if not path.is_file():
            failures.append(f"missing code file: {item['path']}")
            continue
        checked += 1
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            failures.append(f"code identity mismatch: {item['path']}")
    return {"pass": not failures, "checked": checked, "failures": failures}


def _panel_from_cell_rows(
    rows: Iterable[dict[str, Any]],
    *,
    model_key: str,
    stratum_key: str,
    pass_key: str,
    models: list[str],
) -> dict[str, Any]:
    values = {(row[model_key], row[stratum_key]): as_bool(row[pass_key]) for row in rows}
    checkpoint_both = {
        model: values.get((model, "S1_no_contact"), False)
        and values.get((model, "S2_interaction_sensitive"), False)
        for model in models
    }
    stratum_panels = {}
    for stratum in ("S1_no_contact", "S2_interaction_sensitive"):
        count = sum(values.get((model, stratum), False) for model in models)
        stratum_panels[stratum] = {
            "passing_model_count": count,
            "required_model_count": 2,
            "panel_pass": count >= 2,
        }
    count = sum(checkpoint_both.values())
    return {
        "checkpoint_both_strata_pass": checkpoint_both,
        "passing_model_count": count,
        "required_model_count": 2,
        "panel_pass": count >= 2,
        "stratum_panels": stratum_panels,
    }


def reproduce_joint_composition() -> dict[str, Any]:
    verification = verify_public_sources()
    if not verification["pass"]:
        raise RuntimeError("public source verification failed: " + "; ".join(verification["failures"]))
    manifest = read_json(DATA_ROOT / "public_source_manifest.json")
    units = read_csv(DATA_ROOT / "joint_composition_units.csv")
    results = read_csv(DATA_ROOT / "joint_composition_results.csv")
    specificity = read_csv(DATA_ROOT / "joint_specificity_pairs.csv")
    equivalence = read_csv(DATA_ROOT / "affine_composition_equivalence.csv")
    models = manifest["checkpoint_public_labels"]
    thresholds = manifest["metric_thresholds_by_stratum"]

    cell_rows = []
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in results:
        grouped[(row["route"], row["model_id"], row["stratum"])].append(row)
    metric_fields = ["metric_m1_x", "metric_m1_y", "metric_m2", "metric_m3", "metric_m4", "metric_m5"]
    for (route, model, stratum), rows in sorted(grouped.items()):
        medians = {field: statistics.median(float(row[field]) for row in rows) for field in metric_fields}
        coverage = sum(as_bool(row["unit_pass"]) for row in rows) / len(rows)
        threshold_pass = all(
            medians[f"metric_{metric}"] <= float(value)
            for metric, value in thresholds[stratum].items()
        )
        cell_rows.append(
            {
                "route": route,
                "model_id": model,
                "stratum": stratum,
                "unit_count": len(rows),
                "joint_coverage": coverage,
                "median_metrics": medians,
                "cell_pass": coverage >= 0.8 and threshold_pass,
            }
        )
    route_panels = {}
    for route in sorted({row["route"] for row in cell_rows}):
        route_panels[route] = _panel_from_cell_rows(
            [row for row in cell_rows if row["route"] == route],
            model_key="model_id",
            stratum_key="stratum",
            pass_key="cell_pass",
            models=models,
        )

    specificity_panels = {}
    for intended in sorted({row["intended_route"] for row in specificity}):
        specificity_panels[intended] = {}
        for control in sorted({row["control"] for row in specificity if row["intended_route"] == intended}):
            selected = [
                row for row in specificity if row["intended_route"] == intended and row["control"] == control
            ]
            specificity_panels[intended][control] = _panel_from_cell_rows(
                selected,
                model_key="model_id",
                stratum_key="stratum",
                pass_key="pair_pass",
                models=models,
            )

    full_specificity = {}
    for intended in ("addressable_affine_direct", "addressable_affine_composed"):
        checkpoint_pass = {}
        for model in models:
            absolute = next(
                panel_row["cell_pass"]
                for panel_row in cell_rows
                if panel_row["route"] == intended
                and panel_row["model_id"] == model
                and panel_row["stratum"] == "S1_no_contact"
            ) and next(
                panel_row["cell_pass"]
                for panel_row in cell_rows
                if panel_row["route"] == intended
                and panel_row["model_id"] == model
                and panel_row["stratum"] == "S2_interaction_sensitive"
            )
            controls = all(
                all(
                    as_bool(row["pair_pass"])
                    for row in specificity
                    if row["intended_route"] == intended
                    and row["control"] == control
                    and row["model_id"] == model
                )
                for control in ("no_patch", "random_equal_norm", "wrong_object", "matched_wrong_vector")
            )
            checkpoint_pass[model] = absolute and controls
        full_specificity[intended] = {
            "checkpoint_both_strata_absolute_and_all_controls_pass": checkpoint_pass,
            "passing_model_count": sum(checkpoint_pass.values()),
            "required_model_count": 2,
            "panel_pass": sum(checkpoint_pass.values()) >= 2,
        }

    maximum_errors = {
        "coefficient": max(float(row["coefficient_max_abs_error"]) for row in equivalence),
        "patch": max(float(row["patch_max_abs_error"]) for row in equivalence),
        "rollout": max(float(row["rollout_max_abs_error"]) for row in equivalence),
    }
    direct_rows = [row for row in specificity if row["intended_route"] == "addressable_affine_direct"]
    stratum_specificity = {}
    for stratum in ("S1_no_contact", "S2_interaction_sensitive"):
        stratum_specificity[stratum] = {
            control: sum(
                as_bool(row["pair_pass"])
                for row in direct_rows
                if row["stratum"] == stratum and row["control"] == control
            )
            for control in ("no_patch", "random_equal_norm", "wrong_object", "matched_wrong_vector")
        }
    return {
        "schema_version": "1.0",
        "experiment": "joint_velocity_composition",
        "development_only": manifest["development_only"],
        "source_verification": verification,
        "unit_counts": manifest["unit_counts"],
        "rank": manifest["rank"],
        "anchor": manifest["anchor"],
        "autonomous_transitions": manifest["autonomous_transitions"],
        "beta": manifest["beta"],
        "route_panels": route_panels,
        "specificity_panels": specificity_panels,
        "full_specificity": full_specificity,
        "specificity_passing_models_by_stratum": stratum_specificity,
        "affine_equivalence": {
            "maximum_absolute_errors": maximum_errors,
            "coefficient_tolerance": 1e-10,
            "rollout_tolerance": 1e-6,
            "pass": maximum_errors["coefficient"] <= 1e-10
            and maximum_errors["patch"] <= 1e-10
            and maximum_errors["rollout"] <= 1e-6,
        },
        "joint_native_adequacy_supported": route_panels["native_joint"]["panel_pass"],
        "rank4_joint_capacity_supported": route_panels["rank4_oracle"]["panel_pass"],
        "joint_absolute_rollout_supported": route_panels["addressable_affine_direct"]["panel_pass"],
        "joint_full_specificity_established": full_specificity["addressable_affine_direct"]["panel_pass"],
    }


def reproduce_specificity_calibration() -> dict[str, Any]:
    baseline = read_csv(DATA_ROOT / "single_component_specificity_baseline.csv")
    joint = read_csv(DATA_ROOT / "joint_specificity_pairs.csv")
    panels = {}
    for panel in sorted({row["panel"] for row in baseline}):
        models = sorted({row["model_id"] for row in baseline if row["panel"] == panel})
        panels[panel] = {}
        for control in sorted({row["control"] for row in baseline if row["panel"] == panel}):
            selected = [row for row in baseline if row["panel"] == panel and row["control"] == control]
            panels[panel][control] = _panel_from_cell_rows(
                selected,
                model_key="model_id",
                stratum_key="stratum",
                pass_key="pair_pass",
                models=models,
            )
    compatible = all(result["panel_pass"] for panel in panels.values() for result in panel.values())
    comparison = {}
    for control in ("no_patch", "random_equal_norm", "wrong_object"):
        single_values = [
            float(row["dominance_fraction_future"])
            for row in baseline
            if row["panel"] == "development_baseline"
            and row["stratum"] == "S2_interaction_sensitive"
            and row["control"] == control
        ]
        joint_values = [
            float(row["dominance_fraction_future"])
            for row in joint
            if row["intended_route"] == "addressable_affine_direct"
            and row["stratum"] == "S2_interaction_sensitive"
            and row["control"] == control
        ]
        comparison[control] = {
            "single_component_median_future_dominance": statistics.median(single_values),
            "joint_edit_median_future_dominance": statistics.median(joint_values),
            "single_component_stronger": statistics.median(single_values) > statistics.median(joint_values),
        }
    joint_summary = reproduce_joint_composition()
    return {
        "schema_version": "1.0",
        "paired_dominance_threshold": 0.8,
        "single_component_panels": panels,
        "calibrated_threshold_compatible_with_known_positive_baseline": compatible,
        "S2_future_dominance_comparison": comparison,
        "single_component_specificity_baseline_stronger": all(
            value["single_component_stronger"] for value in comparison.values()
        ),
        "joint_full_specificity_established": joint_summary["joint_full_specificity_established"],
        "original_single_component_claim_changed": False,
    }


def reproduce_specificity_layer_localization() -> dict[str, Any]:
    rows = read_csv(DATA_ROOT / "specificity_layer_results.csv")
    decisions = read_csv(DATA_ROOT / "specificity_layer_localization.csv")
    models = sorted({row["model_id"] for row in rows})
    panels: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for control in sorted({row["control"] for row in rows}):
        for stratum in ("S1_no_contact", "S2_interaction_sensitive"):
            for layer in ("native_joint", "rank4_oracle", "addressable_affine"):
                selected = [
                    row
                    for row in rows
                    if row["control"] == control and row["stratum"] == stratum and row["layer"] == layer
                ]
                if not selected:
                    panels[control][stratum][layer] = None
                    continue
                count = sum(as_bool(row["pair_pass"]) for row in selected)
                panels[control][stratum][layer] = {
                    "passing_model_count": count,
                    "required_model_count": 2,
                    "panel_pass": count >= 2,
                }

    reproduced_labels = []
    for row in decisions:
        native = tri_bool(row["native_panel"])
        oracle = tri_bool(row["rank4_oracle_panel"])
        addressable = tri_bool(row["addressable_affine_panel"])
        if native is None:
            if oracle is True and addressable is False:
                label = "loss_at_addressable_operator"
            elif oracle is True and addressable is True:
                label = "no_clear_specificity_loss"
            else:
                label = "mixed_layer_loss"
        elif native is False:
            label = "limited_at_native"
        elif oracle is False:
            label = "loss_at_rank4_projection"
        elif addressable is False:
            label = "loss_at_addressable_operator"
        elif addressable is True:
            label = "no_clear_specificity_loss"
        else:
            label = "mixed_layer_loss"
        if label != row["localization"]:
            raise RuntimeError(f"localization mismatch for {row['control']} {row['stratum']}")
        reproduced_labels.append({"control": row["control"], "stratum": row["stratum"], "label": label})
    unique = {row["label"] for row in reproduced_labels}
    overall = next(iter(unique)) if len(unique) == 1 else "mixed_layer_loss"
    return {
        "schema_version": "1.0",
        "paired_dominance_threshold": 0.8,
        "panels": {control: dict(strata) for control, strata in panels.items()},
        "localization": reproduced_labels,
        "overall_localization": overall,
        "addressable_layer_specificity_loss_observed": any(
            row["label"] == "loss_at_addressable_operator" for row in reproduced_labels
        ),
        "wrong_object_native_ceiling_observed": any(
            row == {
                "control": "wrong_object",
                "stratum": "S2_interaction_sensitive",
                "label": "limited_at_native",
            }
            for row in reproduced_labels
        ),
        "model_count": len(models),
    }


def reproduce_operator_extension() -> dict[str, Any]:
    cells = read_csv(DATA_ROOT / "operator_extension_cells.csv")
    specificity = read_csv(DATA_ROOT / "operator_extension_specificity.csv")
    backward = read_csv(DATA_ROOT / "single_component_backward_compatibility.csv")
    correction = read_csv(DATA_ROOT / "minimal_joint_correction_summary.csv")
    weights = read_json(DATA_ROOT / "operator_weights.json")
    models = sorted({row["model_id"] for row in cells})
    operators = ("original_affine", "joint_aware_affine", "minimal_joint_correction")
    intended_panels = {
        operator: _panel_from_cell_rows(
            [row for row in cells if row["route"] == operator],
            model_key="model_id",
            stratum_key="stratum",
            pass_key="cell_pass",
            models=models,
        )
        for operator in operators
    }
    control_panels = {}
    full_specificity = {}
    for operator in operators:
        control_panels[operator] = {}
        for control in ("no_patch", "random_equal_norm", "wrong_object", "matched_wrong_vector"):
            control_panels[operator][control] = _panel_from_cell_rows(
                [
                    row
                    for row in specificity
                    if row["operator_or_layer"] == operator and row["control"] == control
                ],
                model_key="model_id",
                stratum_key="stratum",
                pass_key="pair_pass",
                models=models,
            )
        checkpoint_pass = {
            model: intended_panels[operator]["checkpoint_both_strata_pass"][model]
            and all(
                control_panels[operator][control]["checkpoint_both_strata_pass"][model]
                for control in control_panels[operator]
            )
            for model in models
        }
        full_specificity[operator] = {
            "checkpoint_both_strata_absolute_and_all_controls_pass": checkpoint_pass,
            "passing_model_count": sum(checkpoint_pass.values()),
            "required_model_count": 2,
            "panel_pass": sum(checkpoint_pass.values()) >= 2,
        }
    backward_panels = {
        operator: _panel_from_cell_rows(
            [row for row in backward if row["operator"] == operator],
            model_key="model_id",
            stratum_key="stratum",
            pass_key="backward_cell_pass",
            models=models,
        )
        for operator in operators
    }
    for operator in operators:
        backward_panels[operator]["random_negative_control_pass"] = all(
            as_bool(row["random_negative_control_panel_pass"])
            for row in backward
            if row["operator"] == operator
        )
    weight_shapes = {
        operator: {
            model: [len(record["weights"]), len(record["weights"][0])]
            for model, record in weights["operators"][operator]["models"].items()
        }
        for operator in operators
    }
    native_wrong_object_s2 = [
        row
        for row in specificity
        if row["operator_or_layer"] == "native_joint"
        and row["control"] == "wrong_object"
        and row["stratum"] == "S2_interaction_sensitive"
    ]
    native_ceiling = sum(as_bool(row["pair_pass"]) for row in native_wrong_object_s2) < 2
    return {
        "schema_version": "1.0",
        "intended_joint_rollout_panels": intended_panels,
        "specificity_panels": control_panels,
        "full_specificity": full_specificity,
        "single_component_backward_compatibility": backward_panels,
        "minimal_joint_correction_by_model_and_stratum": correction,
        "operator_weight_shapes": weight_shapes,
        "selected_operator": "none",
        "original_affine_outcome": "insufficient_specificity",
        "joint_aware_affine_outcome": "insufficient_specificity",
        "minimal_joint_correction_outcome": "insufficient_specificity",
        "joint_aware_affine_closes_gap": full_specificity["joint_aware_affine"]["panel_pass"],
        "minimal_joint_correction_closes_gap": full_specificity["minimal_joint_correction"]["panel_pass"],
        "native_object_specificity_ceiling_observed": native_ceiling,
        "full_joint_specificity_established": any(
            result["panel_pass"] for result in full_specificity.values()
        ),
    }


def summarize_boundary_claim() -> dict[str, Any]:
    joint = reproduce_joint_composition()
    calibration = reproduce_specificity_calibration()
    layers = reproduce_specificity_layer_localization()
    operators = reproduce_operator_extension()
    return {
        "schema_version": "1.0",
        "experiment": "joint_velocity_composition",
        "joint_native_adequacy_supported": joint["joint_native_adequacy_supported"],
        "rank4_joint_capacity_supported": joint["rank4_joint_capacity_supported"],
        "affine_component_composition_exact": joint["affine_equivalence"]["pass"],
        "joint_absolute_rollout_supported": joint["joint_absolute_rollout_supported"],
        "single_component_specificity_baseline_stronger": calibration[
            "single_component_specificity_baseline_stronger"
        ],
        "specificity_calibration_compatible": calibration[
            "calibrated_threshold_compatible_with_known_positive_baseline"
        ],
        "joint_full_specificity_established": joint["joint_full_specificity_established"],
        "addressable_layer_specificity_loss_observed": layers[
            "addressable_layer_specificity_loss_observed"
        ],
        "specificity_layer_localization": layers["overall_localization"],
        "wrong_object_native_ceiling_observed": layers["wrong_object_native_ceiling_observed"],
        "joint_aware_affine_closes_gap": operators["joint_aware_affine_closes_gap"],
        "minimal_joint_correction_closes_gap": operators[
            "minimal_joint_correction_closes_gap"
        ],
        "full_joint_specificity_claim_allowed": joint["joint_full_specificity_established"]
        and operators["full_joint_specificity_established"],
        "development_only": joint["development_only"],
        "primary_single_component_claim_changed": False,
        "carrier_capacity_not_equal_compositional_addressability": joint[
            "rank4_joint_capacity_supported"
        ]
        and not joint["joint_full_specificity_established"],
    }


SUMMARY_FUNCTIONS: dict[str, Callable[[], dict[str, Any]]] = {
    "joint_composition_summary.json": reproduce_joint_composition,
    "specificity_calibration_summary.json": reproduce_specificity_calibration,
    "specificity_layer_summary.json": reproduce_specificity_layer_localization,
    "operator_extension_summary.json": reproduce_operator_extension,
    "boundary_claim_summary.json": summarize_boundary_claim,
}


def reproduce_all_summaries() -> dict[str, Any]:
    outputs = {}
    for filename, function in SUMMARY_FUNCTIONS.items():
        value = function()
        expected = read_json(EXPECTED_ROOT / filename)
        if value != expected:
            raise RuntimeError(f"reproduced summary differs from expected/{filename}")
        target = REPRODUCED_ROOT / filename
        write_json(target, value)
        outputs[filename] = {
            "matches_expected": True,
            "output": target.relative_to(PUBLIC_ROOT).as_posix(),
        }
    return {"status": "complete", "outputs": outputs}


def run_one(filename: str) -> dict[str, Any]:
    value = SUMMARY_FUNCTIONS[filename]()
    expected = read_json(EXPECTED_ROOT / filename)
    if value != expected:
        raise RuntimeError(f"reproduced summary differs from expected/{filename}")
    target = REPRODUCED_ROOT / filename
    write_json(target, value)
    return {
        "status": "complete",
        "matches_expected": True,
        "output": target.relative_to(PUBLIC_ROOT).as_posix(),
        "summary": value,
    }
