"""Render Paper 1 Section 5.2 from publication-facing source CSV files only."""

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


PRIMARY_RANKS = [1, 2, 4, 8, 12, 16, 20, 32]
EXPECTED_ADDRESSABLE_COUNTS = [0, 1, 3, 3, 3, 3, 3, 3]
EXPECTED_RANK4_CAPTURE = [
    0.823883866943921,
    0.8017959074959805,
    0.8655577359146452,
]
SELECTED_RANK = 4

EXPECTED_MODEL_LABELS = ["Development model 1", "Development model 2", "Development model 3"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing figure source data: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"Figure source data is empty: {path}")
    return rows


def load_and_validate_plot_data(source_dir: Path) -> dict[str, Any]:
    counts_path = source_dir / "figure_rank_transition_counts.csv"
    spectrum_path = source_dir / "figure_rank_transition_spectrum.csv"
    count_rows = read_rows(counts_path)
    spectrum_rows = read_rows(spectrum_path)

    ranks = [int(row["rank"]) for row in count_rows]
    require(ranks == PRIMARY_RANKS, f"Unexpected primary rank grid: {ranks}")
    addressable_counts = [int(row["addressable_passing_checkpoints"]) for row in count_rows]
    oracle_counts = [int(row["oracle_passing_checkpoints"]) for row in count_rows]
    panel_requirements = {int(row["panel_requirement"]) for row in count_rows}
    selected_ranks = {int(row["selected_rank"]) for row in count_rows}

    require(
        addressable_counts == EXPECTED_ADDRESSABLE_COUNTS,
        "Source-data addressable pass counts contradict the frozen readback: "
        f"expected {EXPECTED_ADDRESSABLE_COUNTS}, observed {addressable_counts}",
    )
    require(
        oracle_counts == EXPECTED_ADDRESSABLE_COUNTS,
        "Source-data oracle pass counts contradict the frozen readback: "
        f"expected {EXPECTED_ADDRESSABLE_COUNTS}, observed {oracle_counts}",
    )
    require(panel_requirements == {2}, f"Unexpected panel requirement: {panel_requirements}")
    require(selected_ranks == {SELECTED_RANK}, f"Unexpected selected rank: {selected_ranks}")

    spectrum: list[list[float]] = []
    observed_labels: list[str] = []
    for row in spectrum_rows:
        label = row["model_label"]
        if label not in observed_labels:
            observed_labels.append(label)
    require(observed_labels == EXPECTED_MODEL_LABELS, f"Unexpected model labels: {observed_labels}")
    for label in observed_labels:
        model_rows = [row for row in spectrum_rows if row["model_label"] == label]
        model_ranks = [int(row["rank"]) for row in model_rows]
        require(model_ranks == PRIMARY_RANKS, f"Unexpected rank grid for {label}: {model_ranks}")
        ratios = [float(row["cumulative_uncentered_second_moment_ratio"]) for row in model_rows]
        require(
            all(0.0 <= value <= 1.0 for value in ratios),
            f"{label} has a cumulative ratio outside [0,1]",
        )
        require(
            all(left <= right for left, right in zip(ratios, ratios[1:])),
            f"{label} cumulative ratios are not monotone",
        )
        spectrum.append(ratios)

    rank4_index = PRIMARY_RANKS.index(SELECTED_RANK)
    observed_rank4 = [row[rank4_index] for row in spectrum]
    require(
        np.allclose(observed_rank4, EXPECTED_RANK4_CAPTURE, rtol=0.0, atol=1e-15),
        "Source-data rank-4 second-moment ratios contradict the frozen readback: "
        f"expected {EXPECTED_RANK4_CAPTURE}, observed {observed_rank4}",
    )
    rank8_index = PRIMARY_RANKS.index(8)
    observed_rank8 = [row[rank8_index] for row in spectrum]
    require(
        all(0.93 <= value <= 0.97 for value in observed_rank8),
        f"Frozen rank-8 capture lies outside expected 0.93-0.97 range: {observed_rank8}",
    )

    return {
        "primary_ranks": PRIMARY_RANKS,
        "addressable_pass_counts": addressable_counts,
        "oracle_pass_counts": oracle_counts,
        "addressable_selected_rank": SELECTED_RANK,
        "oracle_selected_rank": SELECTED_RANK,
        "cumulative_second_moment_ratios": spectrum,
        "source_files": [counts_path, spectrum_path],
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.6,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def make_figure(data: dict[str, Any]) -> plt.Figure:
    configure_style()
    ranks = data["primary_ranks"]
    x = np.arange(len(ranks), dtype=float)
    selected_x = float(ranks.index(SELECTED_RANK))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.035, h_pad=0.035, wspace=0.08, hspace=0.02)

    # Panel (a): a small symmetric offset keeps coincident frozen curves visible.
    ax = axes[0]
    ax.plot(
        x - 0.035,
        data["addressable_pass_counts"],
        color="#0072B2",
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.25,
        linestyle="-",
        label="Addressable rollout",
        zorder=4,
    )
    ax.plot(
        x + 0.035,
        data["oracle_pass_counts"],
        color="#D55E00",
        marker="s",
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.15,
        linestyle="--",
        label="Oracle capacity",
        zorder=3,
    )
    ax.axhline(2, color="#666666", linewidth=1.0, linestyle=":", zorder=1)
    ax.axvline(selected_x, color="#333333", linewidth=1.0, linestyle="-.", zorder=1)
    ax.text(
        0.98,
        2.07,
        "2-of-3 panel requirement",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=8.2,
    )
    ax.text(
        selected_x + 0.12,
        0.08,
        "selected rank = 4",
        rotation=90,
        ha="left",
        va="bottom",
        color="#333333",
        fontsize=8.2,
    )
    ax.set_title("(a) Smallest passing tested rank", loc="left", pad=7)
    ax.set_xlabel("Tested carrier rank")
    ax.set_ylabel("Passing development checkpoints (of 3)")
    ax.set_xticks(x, [str(rank) for rank in ranks])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlim(-0.35, len(ranks) - 0.65)
    ax.set_ylim(-0.08, 3.32)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.75)
    ax.legend(loc="lower right", frameon=False, handlelength=2.5)

    # Panel (b): labels are intentionally checkpoint-identity-free.
    ax = axes[1]
    styles = [
        ("#0072B2", "o", "-"),
        ("#D55E00", "s", "--"),
        ("#009E73", "^", ":"),
    ]
    for index, (ratios, style) in enumerate(
        zip(data["cumulative_second_moment_ratios"], styles), start=1
    ):
        color, marker, linestyle = style
        ax.plot(
            x,
            ratios,
            color=color,
            marker=marker,
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.05,
            linestyle=linestyle,
            label=f"Development model {index}",
            zorder=3,
        )
    ax.axvline(selected_x, color="#333333", linewidth=1.0, linestyle="-.", zorder=1)
    ax.text(
        selected_x + 0.12,
        0.06,
        "rank 4",
        rotation=90,
        ha="left",
        va="bottom",
        color="#333333",
        fontsize=8.2,
    )
    ax.set_title("(b) Carrier spectrum", loc="left", pad=7)
    ax.set_xlabel("Carrier rank")
    ax.set_ylabel("Cumulative uncentered\nsecond-moment ratio")
    ax.set_xticks(x, [str(rank) for rank in ranks])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlim(-0.35, len(ranks) - 0.65)
    ax.set_ylim(0.0, 1.03)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.75)
    ax.legend(loc="lower right", frameon=False, handlelength=2.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.0, width=0.8)

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

    plot_data = load_and_validate_plot_data(source_dir)
    figure = make_figure(plot_data)

    png_path = output_dir / "figure_rank_transition_and_spectrum.png"
    pdf_path = output_dir / "figure_rank_transition_and_spectrum.pdf"
    figure.savefig(
        png_path,
        dpi=600,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Development rank transition and carrier spectrum",
            "Software": f"Matplotlib {matplotlib.__version__}",
        },
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "Development rank transition and carrier spectrum",
            "Subject": "Paper 1 Section 5.2 manuscript figure",
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
        "source_data_hashes": {str(path): sha256(path) for path in plot_data["source_files"]},
        "primary_ranks": plot_data["primary_ranks"],
        "addressable_pass_counts": plot_data["addressable_pass_counts"],
        "oracle_pass_counts": plot_data["oracle_pass_counts"],
        "rank4_capture": [row[2] for row in plot_data["cumulative_second_moment_ratios"]],
        "outputs": [str(png_path), str(pdf_path)],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
