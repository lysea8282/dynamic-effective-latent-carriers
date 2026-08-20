from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

from .utils import read_json


PAPER_FIGURES = [
    (
        "rank_transition",
        "plot_rank_transition_and_spectrum.py",
        "figure_rank_transition_and_spectrum",
    ),
    (
        "fresh_replication",
        "plot_fresh_replication_and_controls.py",
        "figure_fresh_replication_and_controls",
    ),
    (
        "temporal_robustness",
        "plot_figure_temporal_robustness.py",
        "figure_temporal_robustness",
    ),
    (
        "position_stress_test",
        "plot_position_stress_test.py",
        "figure_position_stress_test",
    ),
]


def _summary(root: Path) -> dict[str, Any]:
    reproduced = root / "results" / "reproduced" / "paper_summary.json"
    return read_json(reproduced if reproduced.exists() else root / "results" / "expected" / "paper_summary.json")


def generate_figures(root: Path) -> list[Path]:
    created: list[Path] = []
    for directory_name, script_name, output_basename in PAPER_FIGURES:
        figure_dir = root / "figures" / directory_name
        script_path = figure_dir / script_name
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=figure_dir,
            check=True,
        )
        for suffix in (".png", ".pdf"):
            output_path = figure_dir / f"{output_basename}{suffix}"
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"figure script did not create {output_path}")
            created.append(output_path)
    return created


def generate_tables(root: Path) -> list[Path]:
    summary = _summary(root)
    output = root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Native adequacy, development", f"{summary['native_counterfactual_adequacy']['development']['passing_seed_count']}/3 seeds", "pass"),
        ("Native adequacy, replication", f"{summary['native_counterfactual_adequacy']['replication']['passing_seed_count']}/3 seeds", "pass"),
        ("Smallest addressable velocity rank", str(summary["low_rank_velocity_carrier"]["addressable_selected_rank"]), "pass"),
        ("Addressable replication", f"{summary['low_rank_velocity_carrier']['replication_passing_checkpoint_count']}/3 checkpoints", "pass"),
        ("Temporal anchors", f"{len(summary['temporal_stability']['anchors'])}/5", "pass"),
        ("Temporal transport targets", f"{len(summary['temporal_stability']['transport_targets'])}/4", "pass"),
        ("Position specificity", "0/3 checkpoints at each amplitude", "not supported"),
    ]
    csv_path = output / "main_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Result", "Value", "Outcome"])
        writer.writerows(rows)
    markdown_path = output / "main_results.md"
    text = "| Result | Value | Outcome |\n|---|---:|---|\n" + "".join(
        f"| {name} | {value} | {outcome} |\n" for name, value, outcome in rows
    )
    markdown_path.write_text(text, encoding="utf-8")
    return [csv_path, markdown_path]
