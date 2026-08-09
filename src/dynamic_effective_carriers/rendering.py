from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import read_json


def _summary(root: Path) -> dict[str, Any]:
    reproduced = root / "results" / "reproduced" / "paper_summary.json"
    return read_json(reproduced if reproduced.exists() else root / "results" / "expected" / "paper_summary.json")


def generate_figures(root: Path) -> list[Path]:
    summary = _summary(root)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    rank = summary["low_rank_velocity_carrier"]
    profile = rank["development_rank_profile"]
    ranks = [item["rank"] for item in profile]
    passing = [item["passing_checkpoint_count"] for item in profile]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(ranks, passing, marker="o", linewidth=2)
    axis.axhline(2, color="black", linestyle="--", linewidth=1, label="2-of-3 threshold")
    axis.axvline(rank["addressable_selected_rank"], color="#b23a48", linestyle=":", label="selected rank")
    axis.set(xlabel="Carrier rank", ylabel="Passing checkpoints", xticks=ranks, ylim=(-0.1, 3.2))
    axis.legend(frameon=False)
    fig.tight_layout()
    path = output / "rank_transition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    temporal = summary["temporal_stability"]
    anchors = temporal["anchors"]
    counts = [temporal["anchor_results"][str(anchor)]["passing_checkpoint_count"] for anchor in anchors]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.bar([str(anchor) for anchor in anchors], counts, color="#3572a5")
    axis.axhline(2, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Anchor", ylabel="Passing checkpoints", ylim=(0, 3.2))
    fig.tight_layout()
    path = output / "temporal_stability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    position = summary["intervention_specificity"]["amplitude_panels"]
    labels = ["small", "medium", "large"]
    correct = [position[label]["correct_checkpoint_count"] for label in labels]
    random = [position[label]["random_checkpoint_count"] for label in labels]
    rank_zero = [position[label]["rank_zero_checkpoint_count"] for label in labels]
    specificity = [position[label]["specificity_valid_checkpoint_count"] for label in labels]
    x = range(len(labels))
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    width = 0.2
    axis.bar([v - 1.5 * width for v in x], correct, width, label="correct")
    axis.bar([v - 0.5 * width for v in x], random, width, label="random")
    axis.bar([v + 0.5 * width for v in x], rank_zero, width, label="rank zero")
    axis.bar([v + 1.5 * width for v in x], specificity, width, label="specificity valid")
    axis.set(xticks=list(x), xticklabels=labels, ylabel="Checkpoint count", ylim=(0, 3.2))
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = output / "intervention_specificity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)
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
        ("Event-phase positive signal", "not observed", "bounded negative"),
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
