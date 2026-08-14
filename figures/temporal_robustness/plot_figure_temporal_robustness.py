"""Render Paper 1 Section 5.4 from the publication-facing temporal CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSV_NAME = "figure_temporal_robustness_data.csv"
PNG_NAME = "figure_temporal_robustness.png"
PDF_NAME = "figure_temporal_robustness.pdf"
B1_ASSAY = "B1_per_anchor_existence"
B2_ASSAY = "B2_frozen_t7_transport"
B1_ANCHORS = [5, 6, 7, 8, 9]
B2_TARGETS = [5, 6, 8, 9]
SOURCE_ANCHOR = 7
PUBLIC_MODELS = ["Fresh model 1", "Fresh model 2", "Fresh model 3"]
EXPECTED_FIELDS = [
    "assay",
    "anchor",
    "public_model",
    "S1_pass",
    "S2_pass",
    "checkpoint_pass",
    "S1_joint_coverage",
    "S2_joint_coverage",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_bool(value: str, field: str) -> bool:
    require(value in {"true", "false"}, f"{field} must be 'true' or 'false', observed {value!r}")
    return value == "true"


def load_and_validate(source_path: Path) -> dict[str, Any]:
    require(source_path.is_file(), f"Missing source CSV: {source_path}")
    raw_text = source_path.read_text(encoding="utf-8")
    workflow_labels = ("W" + "31_", "work" + "_items")
    for forbidden in ("291301", "291302", "291303", *workflow_labels):
        require(forbidden not in raw_text, f"Public source CSV contains forbidden internal token: {forbidden}")

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == EXPECTED_FIELDS, f"Unexpected CSV fields: {reader.fieldnames}")
        raw_rows = list(reader)
    require(len(raw_rows) == 27, f"Expected 27 public rows, observed {len(raw_rows)}")

    rows: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, int, str]] = set()
    for raw in raw_rows:
        assay = raw["assay"]
        anchor = int(raw["anchor"])
        model = raw["public_model"]
        require(assay in {B1_ASSAY, B2_ASSAY}, f"Unexpected assay: {assay}")
        require(model in PUBLIC_MODELS, f"Unexpected public model label: {model}")
        if assay == B1_ASSAY:
            require(anchor in B1_ANCHORS, f"Unexpected B1 anchor: {anchor}")
        else:
            require(anchor in B2_TARGETS, f"Unexpected B2 target: {anchor}")
            require(anchor != SOURCE_ANCHOR, "B2 source anchor must not be encoded as a target result")

        key = (assay, anchor, model)
        require(key not in observed_keys, f"Duplicate public row: {key}")
        observed_keys.add(key)
        s1_pass = parse_bool(raw["S1_pass"], "S1_pass")
        s2_pass = parse_bool(raw["S2_pass"], "S2_pass")
        checkpoint_pass = parse_bool(raw["checkpoint_pass"], "checkpoint_pass")
        require(checkpoint_pass == (s1_pass and s2_pass), f"Checkpoint pass contradiction: {key}")
        s1_coverage = float(raw["S1_joint_coverage"])
        s2_coverage = float(raw["S2_joint_coverage"])
        require(0.0 <= s1_coverage <= 1.0, f"S1 coverage outside [0,1]: {key}")
        require(0.0 <= s2_coverage <= 1.0, f"S2 coverage outside [0,1]: {key}")
        rows.append(
            {
                "assay": assay,
                "anchor": anchor,
                "public_model": model,
                "S1_pass": s1_pass,
                "S2_pass": s2_pass,
                "checkpoint_pass": checkpoint_pass,
                "S1_joint_coverage": s1_coverage,
                "S2_joint_coverage": s2_coverage,
            }
        )

    expected_keys = {
        (B1_ASSAY, anchor, model) for anchor in B1_ANCHORS for model in PUBLIC_MODELS
    } | {
        (B2_ASSAY, anchor, model) for anchor in B2_TARGETS for model in PUBLIC_MODELS
    }
    require(observed_keys == expected_keys, "Public CSV does not contain the exact registered assay grid")

    counts: dict[str, dict[int, int]] = defaultdict(dict)
    for assay, anchors in ((B1_ASSAY, B1_ANCHORS), (B2_ASSAY, B2_TARGETS)):
        for anchor in anchors:
            anchor_rows = [row for row in rows if row["assay"] == assay and row["anchor"] == anchor]
            require(len(anchor_rows) == 3, f"Expected three models for {assay}, t={anchor}")
            count = sum(int(row["checkpoint_pass"]) for row in anchor_rows)
            require(0 <= count <= 3, f"Passing count outside [0,3] for {assay}, t={anchor}")
            require(count == 2, f"Frozen panel readback expected 2/3 for {assay}, t={anchor}; observed {count}/3")
            counts[assay][anchor] = count

    return {"rows": rows, "counts": counts}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9.5,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(4.55, 9.45)
    ax.set_ylim(0.0, 3.25)
    ax.set_xticks(B1_ANCHORS, [str(anchor) for anchor in B1_ANCHORS])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Anchor t", labelpad=5)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.75, zorder=0)
    threshold_line = ax.axhline(
        2, color="#666666", linewidth=1.05, linestyle=(0, (4, 3)), zorder=1
    )
    ax.legend(
        handles=[threshold_line],
        labels=["2-of-3 panel requirement"],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.93),
        frameon=False,
        fontsize=7.8,
        handlelength=2.5,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)


def panel_heading(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.text(
        0.0,
        1.125,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="semibold",
    )
    ax.text(
        0.0,
        1.045,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#555555",
    )


def build_figure(data: dict[str, Any]) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.18, top=0.78, wspace=0.13)
    fig.supylabel("Fresh models passing both strata (of 3)", x=0.015, fontsize=9.8)

    ax = axes[0]
    style_axis(ax)
    panel_heading(ax, "(a) Per-anchor rank-4 existence", "anchor-specific fit")
    b1_counts = [data["counts"][B1_ASSAY][anchor] for anchor in B1_ANCHORS]
    ax.bar(
        B1_ANCHORS,
        b1_counts,
        width=0.56,
        color="#56B4E9",
        edgecolor="#005A8D",
        linewidth=1.0,
        hatch="///",
        zorder=3,
    )
    ax.plot(
        B1_ANCHORS,
        b1_counts,
        linestyle="none",
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor="#005A8D",
        markeredgewidth=1.1,
        zorder=4,
    )
    for anchor, count in zip(B1_ANCHORS, b1_counts):
        ax.text(anchor, count + 0.12, f"{count}/3", ha="center", va="bottom", fontsize=8.5)

    ax = axes[1]
    style_axis(ax)
    panel_heading(ax, "(b) Frozen-t=7 transport", "same carrier/operator; no refit")
    b2_counts = [data["counts"][B2_ASSAY][anchor] for anchor in B2_TARGETS]
    ax.bar(
        B2_TARGETS,
        b2_counts,
        width=0.56,
        color="#009E73",
        edgecolor="#00684C",
        linewidth=1.0,
        hatch="..",
        zorder=3,
    )
    ax.plot(
        B2_TARGETS,
        b2_counts,
        linestyle="none",
        marker="s",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor="#00684C",
        markeredgewidth=1.1,
        zorder=4,
    )
    for anchor, count in zip(B2_TARGETS, b2_counts):
        ax.text(anchor, count + 0.12, f"{count}/3", ha="center", va="bottom", fontsize=8.5)

    ax.axvspan(6.82, 7.18, color="#B8B8B8", alpha=0.22, linewidth=0, zorder=0)
    ax.axvline(SOURCE_ANCHOR, color="#555555", linewidth=1.0, linestyle=(0, (2, 2)), zorder=2)
    ax.scatter(
        [SOURCE_ANCHOR],
        [0.28],
        marker="D",
        s=31,
        facecolor="white",
        edgecolor="#444444",
        linewidth=1.0,
        zorder=4,
    )
    ax.text(
        SOURCE_ANCHOR,
        0.48,
        "source\nt = 7",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#444444",
        linespacing=1.0,
    )
    return fig


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=script_dir)
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / CSV_NAME
    data = load_and_validate(source_path)
    figure = build_figure(data)

    png_path = output_dir / PNG_NAME
    pdf_path = output_dir / PDF_NAME
    figure.savefig(
        png_path,
        dpi=600,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Temporal robustness of rank-4 effective-state carriers",
            "Software": f"Matplotlib {matplotlib.__version__}",
        },
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Temporal robustness of rank-4 effective-state carriers",
            "Subject": "Paper 1 Section 5.4 manuscript figure",
            "Creator": f"Matplotlib {matplotlib.__version__}",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)

    summary = {
        "status": "FIGURE_GENERATED",
        "sys_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "source_csv_sha256": sha256(source_path),
        "B1_counts": data["counts"][B1_ASSAY],
        "B2_counts": data["counts"][B2_ASSAY],
        "source_anchor": SOURCE_ANCHOR,
        "outputs": [str(png_path), str(pdf_path)],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
