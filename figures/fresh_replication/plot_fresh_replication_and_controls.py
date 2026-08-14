"""Render Paper 1 Figure 2 from publication-facing source CSV files only."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_MODELS = ["Fresh model 1", "Fresh model 2", "Fresh model 3"]
EXPECTED_STRATA = ["S1: non-contact", "S2: interaction-sensitive"]
EXPECTED_COVERAGE = {
    ("Fresh model 1", "S1: non-contact"): 0.8671875,
    ("Fresh model 1", "S2: interaction-sensitive"): 0.9140625,
    ("Fresh model 2", "S1: non-contact"): 1.0,
    ("Fresh model 2", "S2: interaction-sensitive"): 0.921875,
    ("Fresh model 3", "S1: non-contact"): 0.7109375,
    ("Fresh model 3", "S2: interaction-sensitive"): 0.97265625,
}
EXPECTED_MODEL_PASS = {
    "Fresh model 1": True,
    "Fresh model 2": True,
    "Fresh model 3": False,
}
THRESHOLD = 0.80


@dataclass(frozen=True)
class CoverageRow:
    model: str
    stratum: str
    coverage: float
    threshold: float
    cell_pass: bool
    model_pass: bool


@dataclass(frozen=True)
class ControlRow:
    label: str
    aggregation_level: str
    criterion_satisfied_n: int
    denominator: int
    observed_target_pass_n: int | None
    note: str


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected true/false, observed {value!r}")
    return normalized == "true"


def read_coverage(path: Path) -> list[CoverageRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            CoverageRow(
                model=row["model_label"],
                stratum=row["stratum"],
                coverage=float(row["coverage"]),
                threshold=float(row["threshold"]),
                cell_pass=parse_bool(row["cell_pass"]),
                model_pass=parse_bool(row["model_pass"]),
            )
            for row in csv.DictReader(handle)
        ]
    if len(rows) != 6:
        raise ValueError(f"Expected six coverage rows, observed {len(rows)}")

    keyed = {(row.model, row.stratum): row for row in rows}
    if set(keyed) != set(EXPECTED_COVERAGE):
        raise ValueError("Coverage CSV model/stratum support differs from the frozen public support")
    for key, expected in EXPECTED_COVERAGE.items():
        row = keyed[key]
        if not np.isclose(row.coverage, expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"Coverage mismatch for {key}: {row.coverage} != {expected}")
        if not np.isclose(row.threshold, THRESHOLD, rtol=0.0, atol=1e-12):
            raise ValueError(f"Threshold mismatch for {key}: {row.threshold}")
        if row.cell_pass != (row.coverage >= row.threshold):
            raise ValueError(f"Cell-pass value is inconsistent with coverage for {key}")
        if row.model_pass != EXPECTED_MODEL_PASS[row.model]:
            raise ValueError(f"Model-pass value is inconsistent for {row.model}")
    return rows


def read_controls(path: Path) -> list[ControlRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            target_pass = row["observed_target_pass_n"].strip()
            rows.append(
                ControlRow(
                    label=row["control_label"],
                    aggregation_level=row["aggregation_level"],
                    criterion_satisfied_n=int(row["criterion_satisfied_n"]),
                    denominator=int(row["denominator"]),
                    observed_target_pass_n=int(target_pass) if target_pass else None,
                    note=row["outcome_note"],
                )
            )
    if len(rows) != 8:
        raise ValueError(f"Expected eight control-summary rows, observed {len(rows)}")
    for row in rows[:-1]:
        if not (row.criterion_satisfied_n == row.denominator == 6):
            raise ValueError(f"Cell-level control is not 6/6: {row.label}")
    random_row = rows[-1]
    if random_row.label != "Random orthogonal equal norm":
        raise ValueError("Final control row is not the registered random-orthogonal negative control")
    if (random_row.criterion_satisfied_n, random_row.denominator, random_row.observed_target_pass_n) != (
        3,
        3,
        0,
    ):
        raise ValueError("Random-orthogonal control does not encode 3/3 correct behavior and 0/3 target pass")
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def build_figure(coverage_rows: list[CoverageRow], control_rows: list[ControlRow]) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.65),
        gridspec_kw={"width_ratios": [1.04, 0.96]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.035, h_pad=0.035, wspace=0.075, hspace=0.02)

    # Panel (a): grouped bars preserve the failed cell and the full heterogeneity.
    ax = axes[0]
    keyed = {(row.model, row.stratum): row for row in coverage_rows}
    x = np.arange(len(EXPECTED_MODELS), dtype=float)
    width = 0.34
    colors = ["#0072B2", "#E69F00"]
    hatches = ["///", "\\\\"]
    for index, stratum in enumerate(EXPECTED_STRATA):
        values = [keyed[(model, stratum)].coverage for model in EXPECTED_MODELS]
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            color=colors[index],
            edgecolor="#222222",
            linewidth=0.65,
            hatch=hatches[index],
            label=stratum,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#222222",
            )

    ax.axhline(THRESHOLD, color="#555555", linestyle="--", linewidth=1.15, zorder=2)
    ax.text(
        0.0,
        1.002,
        "Dashed line: registered requirement = 0.80",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#4A4A4A",
        fontsize=7.2,
    )
    for position, model in zip(x, EXPECTED_MODELS):
        passed = EXPECTED_MODEL_PASS[model]
        ax.text(
            position,
            1.072,
            "PASS" if passed else "FAIL",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#007A4D" if passed else "#B2182B",
        )
    ax.set_title("(a) Fresh-model replication", loc="left", y=1.145, pad=0)
    ax.set_ylabel("Joint unit coverage")
    ax.set_xticks(x, EXPECTED_MODELS)
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.75, zorder=0)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.065),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
        fontsize=7.8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)

    # Panel (b): a graphical, table-like summary avoids treating the desired
    # random-direction rejection as a failure of the principal method.
    ax = axes[1]
    ax.set_title("(b) Specificity-control outcomes", loc="left", y=1.145, pad=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.15, len(control_rows) + 0.25)
    ax.axis("off")
    ax.text(0.01, len(control_rows) - 0.05, "Control", fontsize=8.1, fontweight="bold", va="bottom")
    ax.text(
        0.99,
        len(control_rows) - 0.05,
        "Fresh-panel outcome",
        fontsize=8.1,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    for index, row in enumerate(control_rows):
        y = len(control_rows) - 1.0 - index
        if index % 2 == 0:
            ax.axhspan(y - 0.42, y + 0.42, color="#F3F3F3", zorder=0)
        ax.text(0.01, y, row.label, ha="left", va="center", fontsize=8.15, color="#222222")
        is_random = row.observed_target_pass_n is not None
        if is_random:
            ax.scatter(
                [0.69],
                [y],
                s=38,
                marker="D",
                facecolor="white",
                edgecolor="#7B3294",
                linewidth=1.3,
                zorder=3,
            )
            outcome = f"{row.observed_target_pass_n}/{row.denominator} target pass"
            outcome_color = "#7B3294"
        else:
            ax.scatter(
                [0.69],
                [y],
                s=40,
                marker="o",
                facecolor="#009E73",
                edgecolor="#006B4F",
                linewidth=0.7,
                zorder=3,
            )
            outcome = f"{row.criterion_satisfied_n}/{row.denominator} cells"
            outcome_color = "#006B4F"
        ax.text(0.99, y, outcome, ha="right", va="center", fontsize=8.15, color=outcome_color)

    ax.text(
        0.01,
        -0.72,
        "Criterion satisfied = registered control behavior;\nnot reproduction of the target counterfactual.",
        ha="left",
        va="top",
        fontsize=7.25,
        color="#555555",
        linespacing=1.15,
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

    coverage_rows = read_coverage(source_dir / "figure_fresh_replication_coverage.csv")
    control_rows = read_controls(source_dir / "figure_fresh_replication_controls.csv")
    figure = build_figure(coverage_rows, control_rows)

    png_path = output_dir / "figure_fresh_replication_and_controls.png"
    pdf_path = output_dir / "figure_fresh_replication_and_controls.pdf"
    figure.savefig(
        png_path,
        dpi=600,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Fresh-model replication and specificity controls",
            "Software": f"Matplotlib {matplotlib.__version__}",
        },
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Fresh-model replication and specificity controls",
            "Subject": "Paper 1 Section 5.3 manuscript figure",
            "Creator": f"Matplotlib {matplotlib.__version__}",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    print(f"WROTE {png_path}")
    print(f"WROTE {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
