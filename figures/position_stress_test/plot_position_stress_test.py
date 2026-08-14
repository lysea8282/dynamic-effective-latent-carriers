"""Render the Paper 1 Section 5.5 position-edit stress-test figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


CSV_NAME = "figure_position_stress_test_source.csv"
PNG_NAME = "figure_position_stress_test.png"
PDF_NAME = "figure_position_stress_test.pdf"
AMPLITUDES = [0.05, 0.10, 0.20]
RAW_CONDITIONS = ["Position patch", "Rank-zero / no patch", "Random equal-norm"]
MATRIX_CONDITIONS = RAW_CONDITIONS + ["Wrong-object", "Full specificity"]
EXPECTED_FIELDS = [
    "panel",
    "amplitude",
    "condition",
    "criterion",
    "passing_models",
    "total_models",
    "outcome",
    "source_file",
    "source_field",
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


def load_and_validate(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing public source CSV: {path}")
    raw_text = path.read_text(encoding="utf-8")
    workflow_labels = tuple("W" + str(number) for number in (29, 30, 31)) + (
        "work" + "_items",
    )
    for token in ("291101", "291102", "291103", *workflow_labels):
        require(token not in raw_text, f"Public source CSV contains forbidden internal token: {token}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == EXPECTED_FIELDS, f"Unexpected CSV fields: {reader.fieldnames}")
        raw_rows = list(reader)
    require(len(raw_rows) == 24, f"Expected 24 plotted rows, observed {len(raw_rows)}")

    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, float, str]] = set()
    for raw in raw_rows:
        panel = raw["panel"]
        amplitude = float(raw["amplitude"])
        condition = raw["condition"]
        criterion = raw["criterion"]
        count = int(raw["passing_models"])
        total = int(raw["total_models"])
        outcome = raw["outcome"]
        require(panel in {"a", "b"}, f"Unexpected panel: {panel}")
        require(amplitude in AMPLITUDES, f"Unexpected amplitude: {amplitude}")
        allowed_conditions = RAW_CONDITIONS if panel == "a" else MATRIX_CONDITIONS
        require(condition in allowed_conditions, f"Unexpected condition for panel {panel}: {condition}")
        key = (panel, amplitude, condition)
        require(key not in keys, f"Duplicate plotted datum: {key}")
        keys.add(key)
        require(total == 3, f"Unexpected denominator for {key}: {total}")
        require(0 <= count <= total, f"Count outside [0,3] for {key}: {count}")
        require(outcome in {"PASS", "FAIL"}, f"Unexpected outcome for {key}: {outcome}")

        if condition in RAW_CONDITIONS:
            require(criterion == "raw_both_strata_gate", f"Unexpected raw criterion for {key}")
            require(count == 3 and outcome == "PASS", f"Frozen raw-gate readback mismatch for {key}")
        elif condition == "Wrong-object":
            require(criterion == "wrong_object_specificity", "Wrong-object criterion mismatch")
            require(count == 0 and outcome == "FAIL", f"Wrong-object readback mismatch for {key}")
        else:
            require(criterion == "all_specificity_controls_valid", "Full-specificity criterion mismatch")
            require(count == 0 and outcome == "FAIL", f"Full-specificity readback mismatch for {key}")

        rows.append(
            {
                "panel": panel,
                "amplitude": amplitude,
                "condition": condition,
                "criterion": criterion,
                "passing_models": count,
                "total_models": total,
                "outcome": outcome,
            }
        )

    expected = {
        ("a", amplitude, condition)
        for amplitude in AMPLITUDES
        for condition in RAW_CONDITIONS
    } | {
        ("b", amplitude, condition)
        for amplitude in AMPLITUDES
        for condition in MATRIX_CONDITIONS
    }
    require(keys == expected, "CSV does not contain the exact registered plot grid")
    return rows


def value(rows: list[dict[str, Any]], panel: str, amplitude: float, condition: str) -> int:
    matches = [
        row["passing_models"]
        for row in rows
        if row["panel"] == panel
        and row["amplitude"] == amplitude
        and row["condition"] == condition
    ]
    require(len(matches) == 1, f"Expected one value for {panel}, {amplitude}, {condition}")
    return int(matches[0])


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9.3,
            "axes.labelsize": 9.7,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 8.6,
            "axes.linewidth": 0.8,
            "legend.fontsize": 7.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_heading(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.text(
        0.0,
        1.13,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.7,
        fontweight="semibold",
    )
    ax.text(
        0.0,
        1.05,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#555555",
    )


def build_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 4.2),
        gridspec_kw={"width_ratios": [1.05, 1.12]},
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.235, top=0.78, wspace=0.43)

    # Panel (a): the same raw gate is satisfied by intended and negative-control routes.
    ax = axes[0]
    panel_heading(
        ax,
        "(a) Raw success is not control-specific",
        "all three routes satisfy the same raw gate",
    )
    x = np.arange(len(AMPLITUDES), dtype=float)
    width = 0.23
    styles = [
        ("#56B4E9", "#005A8D", "///", "o"),
        ("#B8B8B8", "#555555", "\\\\", "s"),
        ("#E69F00", "#9A5D00", "xx", "^")
    ]
    offsets = [-width, 0.0, width]
    handles = []
    for condition, offset, style in zip(RAW_CONDITIONS, offsets, styles):
        face, edge, hatch, marker = style
        counts = [value(rows, "a", amplitude, condition) for amplitude in AMPLITUDES]
        bars = ax.bar(
            x + offset,
            counts,
            width=width * 0.90,
            color=face,
            edgecolor=edge,
            linewidth=0.9,
            hatch=hatch,
            label=condition,
            zorder=3,
        )
        handles.append(bars)
        ax.plot(
            x + offset,
            counts,
            linestyle="none",
            marker=marker,
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=edge,
            markeredgewidth=1.0,
            zorder=4,
        )
        for xpos, count in zip(x + offset, counts):
            ax.text(xpos, count + 0.10, f"{count}/3", ha="center", va="bottom", fontsize=7.4)

    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0.0, 4.25)
    ax.set_xticks(x, [f"{amplitude:.2f}" for amplitude in AMPLITUDES])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Absolute position amplitude", labelpad=5)
    ax.set_ylabel("Models passing the raw gate (of 3)")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.75, zorder=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)

    # Panel (b): exact frozen pass-count matrix; text and shapes avoid color-only encoding.
    ax = axes[1]
    panel_heading(
        ax,
        "(b) Specificity does not isolate the patch",
        "registered outcomes across tested amplitudes",
    )
    row_labels = [
        "Position patch\n(raw gate)",
        "Rank-zero / no patch\n(raw gate)",
        "Random equal-norm\n(raw gate)",
        "Wrong-object\n(specificity)",
        "Full specificity\n(all controls)",
    ]
    for row_index, condition in enumerate(MATRIX_CONDITIONS):
        for col_index, amplitude in enumerate(AMPLITUDES):
            count = value(rows, "b", amplitude, condition)
            passed = count >= 2
            face = "#BDE4F5" if passed else "#F6D29A"
            edge = "#006A9C" if passed else "#9A5D00"
            hatch = "///" if passed else "xx"
            ax.add_patch(
                Rectangle(
                    (col_index - 0.37, row_index - 0.36),
                    0.74,
                    0.72,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.9,
                    hatch=hatch,
                    zorder=2,
                )
            )
            marker = "o" if passed else "X"
            ax.scatter(
                [col_index],
                [row_index - 0.09],
                marker=marker,
                s=24 if passed else 31,
                facecolor="white" if passed else edge,
                edgecolor=edge,
                linewidth=1.0,
                zorder=3,
            )
            ax.text(
                col_index,
                row_index + 0.16,
                f"{count}/3  {'PASS' if passed else 'FAIL'}",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="semibold",
                color="#222222",
                zorder=4,
            )

    ax.axhline(2.5, color="#777777", linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(4.55, -0.55)
    ax.set_xticks(np.arange(3), [f"{amplitude:.2f}" for amplitude in AMPLITUDES])
    ax.set_yticks(np.arange(5), row_labels)
    ax.set_xlabel("Absolute position amplitude", labelpad=5)
    ax.tick_params(axis="x", direction="out", length=3.0, width=0.8)
    ax.tick_params(axis="y", length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

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
    rows = load_and_validate(source_path)
    figure = build_figure(rows)

    png_path = output_dir / PNG_NAME
    pdf_path = output_dir / PDF_NAME
    figure.savefig(
        png_path,
        dpi=600,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Position-edit stress test",
            "Software": f"Matplotlib {matplotlib.__version__}",
        },
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Position-edit stress test",
            "Subject": "Paper 1 Section 5.5 manuscript figure",
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
        "panel_a_counts": {
            condition: [value(rows, "a", amplitude, condition) for amplitude in AMPLITUDES]
            for condition in RAW_CONDITIONS
        },
        "panel_b_wrong_object_counts": [
            value(rows, "b", amplitude, "Wrong-object") for amplitude in AMPLITUDES
        ],
        "panel_b_full_specificity_counts": [
            value(rows, "b", amplitude, "Full specificity") for amplitude in AMPLITUDES
        ],
        "outputs": [str(png_path), str(pdf_path)],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
