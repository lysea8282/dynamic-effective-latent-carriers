from __future__ import annotations

import json
import re
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPT_ROOT = EXPERIMENT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from joint_velocity import (  # noqa: E402
    SUMMARY_FUNCTIONS,
    read_csv,
    read_json,
    reproduce_joint_composition,
    sha256,
    verify_public_sources,
)


def test_public_source_manifest_hashes() -> None:
    result = verify_public_sources()
    assert result["pass"], result["failures"]
    assert result["checked"] >= 19


def test_metric_threshold_registry_is_exact_and_stratum_specific() -> None:
    manifest = read_json(EXPERIMENT_ROOT / "data" / "public_source_manifest.json")
    expected = {
        "S1_no_contact": {
            "m1_x": 0.38875516163789126,
            "m1_y": 0.38875516163789126,
            "m2": 0.6530927312234215,
            "m3": 0.28744727060560543,
            "m4": 0.3393832556356259,
            "m5": 20.260185026563704,
        },
        "S2_interaction_sensitive": {
            "m1_x": 0.4669759655361256,
            "m1_y": 0.4669759655361256,
            "m2": 0.6568131957898696,
            "m3": 0.5513987778513324,
            "m4": 0.25616304846974386,
            "m5": 10.47115975990891,
        },
    }
    assert "metric_thresholds_by_model" not in manifest
    assert manifest["metric_thresholds_by_stratum"] == expected
    assert expected["S1_no_contact"] != expected["S2_interaction_sensitive"]


def test_stratum_threshold_repair_counts_and_boundary_invariants() -> None:
    joint = reproduce_joint_composition()
    expected_s2_counts = {
        "addressable_affine_composed": 3,
        "addressable_affine_direct": 3,
        "full_hidden_equivalence": 3,
        "matched_wrong_vector": 2,
        "native_joint": 3,
        "no_patch": 3,
        "random_equal_norm": 3,
        "rank4_oracle": 3,
        "wrong_object": 1,
    }
    observed = {
        route: panel["stratum_panels"]["S2_interaction_sensitive"]["passing_model_count"]
        for route, panel in joint["route_panels"].items()
    }
    assert observed == expected_s2_counts
    boundary = SUMMARY_FUNCTIONS["boundary_claim_summary.json"]()
    assert boundary["joint_native_adequacy_supported"]
    assert boundary["rank4_joint_capacity_supported"]
    assert boundary["affine_component_composition_exact"]
    assert boundary["joint_absolute_rollout_supported"]
    assert not boundary["joint_full_specificity_established"]
    assert not boundary["joint_aware_affine_closes_gap"]
    assert not boundary["minimal_joint_correction_closes_gap"]
    assert not boundary["full_joint_specificity_claim_allowed"]
    assert boundary["development_only"]


def test_frozen_raw_joint_velocity_inputs_keep_reviewed_hashes() -> None:
    expected_hashes = {
        "affine_composition_equivalence.csv": "7F9CC899A6937CD0BB3F55D75CBBB459FA60AC98A176B0AB1C7A52355BBF1C32",
        "joint_composition_results.csv": "C46C44EDAC0A7A17FC8649C8ED62CFFE7760DCD10410912FFB5414FD932987CE",
        "joint_composition_units.csv": "0FA454981F9D97751F93C2F636BD5C2E9E5002712E31ED851D2A1CB76303D4FC",
        "joint_specificity_pairs.csv": "D763ECF5CD63ABE9637157034EDDBD5619E8A7057EC50FC3BA97DCDF87FBBEF2",
        "minimal_joint_correction_summary.csv": "0DBC38ABDD76DCC56D64F5675653542F54B69A00872114D7474F7C2FE1D456AC",
        "operator_extension_cells.csv": "5EEBAD24E423D0B5F6A55D5F52395104AFD45290B2A2D0E4F120AB4EF820B0C9",
        "operator_extension_specificity.csv": "7795B59D5EB7BF58A0CBC5AEE7E5BCF49175E3598F942822AFB7AFAA14A393C1",
        "operator_fit_diagnostics.csv": "97C213F27B050819B73CEBCAF61181FDDDED3B9DE68594D51B9EBB6D68009D4A",
        "operator_weights.json": "4B3B17584B3CEE374DB8AD8F04435F0E2530DD9909099AB4D97F2D8ED7A9BDDA",
        "single_component_backward_compatibility.csv": "B4A4A7195B0CBD83C639A1556ED7433A174C4391B46692F376D3606F8E67E526",
        "single_component_specificity_baseline.csv": "CD94A9C32182A2B158FBA98F53FC7699C2BBA799C009A9A18499A2B1D3915E6E",
        "specificity_layer_localization.csv": "F6519149C526CC16B6E91C8AF49FDEFFFC4C9004EA8930FF6CD9ACE2B265377F",
        "specificity_layer_results.csv": "33EE5D8EAF4EBAE71545BC2DD6EDC065C287E1B194D63E15889DC3C4A1769604",
    }
    assert {
        name: sha256(EXPERIMENT_ROOT / "data" / name) for name in expected_hashes
    } == expected_hashes


def test_joint_unit_counts_and_balance() -> None:
    rows = read_csv(EXPERIMENT_ROOT / "data" / "joint_composition_units.csv")
    assert len(rows) == 512
    assert sum(row["stratum"] == "S1_no_contact" for row in rows) == 256
    assert sum(row["stratum"] == "S2_interaction_sensitive" for row in rows) == 256
    assert len({row["unit_id"] for row in rows}) == 512


def test_joint_edit_support_and_same_object_contract() -> None:
    manifest = read_json(EXPERIMENT_ROOT / "data" / "public_source_manifest.json")
    rows = read_csv(EXPERIMENT_ROOT / "data" / "joint_composition_units.csv")
    allowed = set(manifest["allowed_absolute_component_magnitudes"])
    for row in rows:
        assert abs(float(row["delta_vx"])) in allowed
        assert abs(float(row["delta_vy"])) in allowed
        assert float(row["delta_v_norm"]) <= float(row["delta_v_norm_max"]) + 1e-12
        assert float(row["delta_v_norm"]) <= manifest["maximum_joint_delta_norm"] + 1e-12
        assert float(row["delta_vx"]) != 0.0
        assert float(row["delta_vy"]) != 0.0
        assert row["both_components_nonzero"] == "true"
        assert row["same_object_edit"] == "true"
        assert row["intended_target"] == "same_object_joint_velocity"


def test_direct_and_composed_affine_equivalence() -> None:
    rows = read_csv(EXPERIMENT_ROOT / "data" / "affine_composition_equivalence.csv")
    assert len(rows) == 512 * 3
    assert max(float(row["coefficient_max_abs_error"]) for row in rows) <= 1e-10
    assert max(float(row["patch_max_abs_error"]) for row in rows) <= 1e-10
    assert max(float(row["rollout_max_abs_error"]) for row in rows) <= 1e-6


def test_frozen_rank_anchor_horizon_and_route_coverage() -> None:
    manifest = read_json(EXPERIMENT_ROOT / "data" / "public_source_manifest.json")
    assert manifest["rank"] == 4
    assert manifest["anchor"] == 7
    assert manifest["autonomous_transitions"] == 12
    results = read_csv(EXPERIMENT_ROOT / "data" / "joint_composition_results.csv")
    assert len(results) == 512 * 3 * 9
    assert {row["autonomous_transition_count"] for row in results} == {"12"}
    assert {row["route"] for row in results} == {
        "native_joint",
        "full_hidden_equivalence",
        "rank4_oracle",
        "addressable_affine_direct",
        "addressable_affine_composed",
        "no_patch",
        "random_equal_norm",
        "wrong_object",
        "matched_wrong_vector",
    }


def test_calibration_and_layer_records_are_complete() -> None:
    baseline = read_csv(EXPERIMENT_ROOT / "data" / "single_component_specificity_baseline.csv")
    assert len(baseline) == 36
    assert {row["panel"] for row in baseline} == {
        "development_baseline",
        "independent_replication_baseline",
    }
    layers = read_csv(EXPERIMENT_ROOT / "data" / "specificity_layer_results.csv")
    assert len(layers) == 66
    assert {row["layer"] for row in layers} == {
        "native_joint",
        "rank4_oracle",
        "addressable_affine",
    }
    localization = read_csv(EXPERIMENT_ROOT / "data" / "specificity_layer_localization.csv")
    assert len(localization) == 8


def test_operator_extension_and_backward_compatibility_records() -> None:
    cells = read_csv(EXPERIMENT_ROOT / "data" / "operator_extension_cells.csv")
    assert len(cells) == 42
    assert {row["route"] for row in cells} >= {
        "original_affine",
        "joint_aware_affine",
        "minimal_joint_correction",
    }
    backward = read_csv(EXPERIMENT_ROOT / "data" / "single_component_backward_compatibility.csv")
    assert len(backward) == 18
    assert {row["operator"] for row in backward} == {
        "original_affine",
        "joint_aware_affine",
        "minimal_joint_correction",
    }
    assert all(row["evaluation_units_used_for_fit"] == "false" for row in backward)
    assert all(row["backward_cell_pass"] == "true" for row in backward)

    weights = read_json(EXPERIMENT_ROOT / "data" / "operator_weights.json")
    expected_features = {
        "original_affine": 13,
        "joint_aware_affine": 13,
        "minimal_joint_correction": 15,
    }
    for operator, feature_count in expected_features.items():
        records = weights["operators"][operator]["models"]
        assert len(records) == 3
        for record in records.values():
            assert len(record["weights"]) == feature_count
            assert {len(row) for row in record["weights"]} == {4}
            assert record["rank"] == 4


def test_reproduced_summaries_match_expected() -> None:
    for filename, function in SUMMARY_FUNCTIONS.items():
        assert function() == read_json(EXPERIMENT_ROOT / "expected" / filename)


def test_boundary_claim_is_bounded() -> None:
    summary = read_json(EXPERIMENT_ROOT / "expected" / "boundary_claim_summary.json")
    assert summary["joint_native_adequacy_supported"]
    assert summary["rank4_joint_capacity_supported"]
    assert summary["affine_component_composition_exact"]
    assert summary["joint_absolute_rollout_supported"]
    assert summary["single_component_specificity_baseline_stronger"]
    assert summary["addressable_layer_specificity_loss_observed"]
    assert not summary["joint_full_specificity_established"]
    assert not summary["joint_aware_affine_closes_gap"]
    assert not summary["minimal_joint_correction_closes_gap"]
    assert not summary["full_joint_specificity_claim_allowed"]
    assert summary["development_only"]
    assert not summary["primary_single_component_claim_changed"]


def test_public_experiment_has_no_private_process_names_or_paths() -> None:
    forbidden = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bV[" + "1-5" + r"]\b",
            r"\bW(?:29|30|31)\b",
            r"developer\s+" + "author" + "ity",
            r"\brole\s+[ab]\b",
            r"formal\s+package",
            "au" + r"dit\s+package",
            r"work" + "_" + "item",
            r"stage2[_-]v" + "3",
        )
    ]
    absolute_path = re.compile(r"\b[A-Za-z]:[\\/]")
    findings = []
    for path in EXPERIMENT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        if absolute_path.search(text) or any(pattern.search(text) for pattern in forbidden):
            findings.append(path.relative_to(EXPERIMENT_ROOT).as_posix())
    assert not findings


def test_expected_files_are_lf_and_manifest_bound_data_are_stable() -> None:
    manifest = read_json(EXPERIMENT_ROOT / "data" / "public_source_manifest.json")
    for item in manifest["files"]:
        path = EXPERIMENT_ROOT / "data" / item["name"]
        assert sha256(path) == item["sha256"]
        if path.suffix in {".csv", ".json"}:
            assert b"\r\n" not in path.read_bytes()
    for path in (EXPERIMENT_ROOT / "expected").glob("*.json"):
        assert b"\r\n" not in path.read_bytes()
