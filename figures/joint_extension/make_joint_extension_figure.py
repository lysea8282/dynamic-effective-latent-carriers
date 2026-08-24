from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FAMILIES = ("M0", "M1", "M2")
COLORS = {"M0": "#8b95a5", "M1": "#2f75b5", "M2": "#d9822b"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def checkpoint_values(table: list[dict[str, str]], family: str, metric: str) -> list[float]:
    values = [
        float(row[metric])
        for row in table
        if row["scope"] == "checkpoint" and row["family"] == family
    ]
    if len(values) != 3:
        raise ValueError(f"expected three checkpoint values for {family}/{metric}")
    return values


def render(output: Path) -> tuple[Path, Path]:
    behavior = rows(HERE / "joint_behavior_panel.csv")
    composition = rows(HERE / "joint_composition_panel.csv")
    addressability = rows(HERE / "joint_addressability_panel.csv")
    if len(behavior) != 12 or len(composition) != 12 or len(addressability) != 6:
        raise ValueError("unexpected corrected Joint figure-source cardinality")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.75), constrained_layout=True)
    x = np.arange(len(FAMILIES), dtype=float)
    offsets = np.array([-0.11, 0.0, 0.11])

    for index, family in enumerate(FAMILIES):
        values = checkpoint_values(behavior, family, "coverage")
        median = float(
            next(row["coverage"] for row in behavior if row["scope"] == "family_median" and row["family"] == family)
        )
        axes[0].scatter(x[index] + offsets, values, color=COLORS[family], s=35, zorder=3)
        axes[0].hlines(median, x[index] - 0.18, x[index] + 0.18, color="black", linewidth=2)
    axes[0].set_title("A  Corrected Joint behavior")
    axes[0].set_ylabel("Joint coverage")
    axes[0].set_ylim(0.82, 1.02)

    for index, family in enumerate(FAMILIES):
        values = checkpoint_values(composition, family, "E_add")
        median = float(
            next(row["E_add"] for row in composition if row["scope"] == "family_median" and row["family"] == family)
        )
        axes[1].scatter(x[index] + offsets, values, color=COLORS[family], s=35, zorder=3)
        axes[1].hlines(median, x[index] - 0.18, x[index] + 0.18, color="black", linewidth=2)
    axes[1].set_title("B  Static composition")
    axes[1].set_ylabel(r"$E_{add}$ (lower is better)")
    axes[1].set_ylim(0.38, 0.98)

    routes = (
        ("Native_coverage", "Native", "#4c78a8"),
        ("U4_oracle_coverage", "U4 oracle", "#72b7b2"),
        ("Addressable_coverage", "Single-only", "#f58518"),
    )
    address_families = ("M1", "M2")
    address_x = np.arange(2, dtype=float)
    for route_index, (field, label, color) in enumerate(routes):
        route_offset = (route_index - 1) * 0.22
        for family_index, family in enumerate(address_families):
            values = [float(row[field]) for row in addressability if row["family"] == family]
            if len(values) != 3:
                raise ValueError(f"expected three addressability checkpoints for {family}/{field}")
            point_offsets = np.array([-0.04, 0.0, 0.04])
            axes[2].scatter(
                address_x[family_index] + route_offset + point_offsets,
                values,
                color=color,
                s=28,
                label=label if family_index == 0 else None,
                zorder=3,
            )
            axes[2].hlines(
                float(np.median(values)),
                address_x[family_index] + route_offset - 0.08,
                address_x[family_index] + route_offset + 0.08,
                color="black",
                linewidth=1.5,
            )
    axes[2].set_title("C  Same-101 addressability")
    axes[2].set_ylabel("Joint coverage")
    axes[2].set_ylim(0.84, 1.02)
    axes[2].set_xticks(address_x, address_families)
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")

    for axis in axes[:2]:
        axis.set_xticks(x, FAMILIES)
    for axis in axes:
        axis.grid(axis="y", color="#d9dde3", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the corrected Joint-extension public draft figure.")
    parser.add_argument("--output", type=Path, default=HERE / "figure_joint_extension")
    args = parser.parse_args()
    for path in render(args.output):
        print(path)


if __name__ == "__main__":
    main()
