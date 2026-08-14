# Paper figure reproduction

The four subdirectories contain the publication-layout plotting script and the public CSV input needed for each main-paper figure. Every script is self-contained, uses paths relative to its own directory by default, and writes both PNG and PDF output beside the script.

| Paper figure | Public source directory | Command from repository root | Generated basename |
|---|---|---|---|
| Figure 1 — rank transition and carrier spectrum | `figures/rank_transition/` | `python figures/rank_transition/plot_rank_transition_and_spectrum.py` | `figure_rank_transition_and_spectrum` |
| Figure 2 — fresh replication and specificity controls | `figures/fresh_replication/` | `python figures/fresh_replication/plot_fresh_replication_and_controls.py` | `figure_fresh_replication_and_controls` |
| Figure 3 — temporal robustness | `figures/temporal_robustness/` | `python figures/temporal_robustness/plot_figure_temporal_robustness.py` | `figure_temporal_robustness` |
| Figure 4 — position-edit stress test | `figures/position_stress_test/` | `python figures/position_stress_test/plot_position_stress_test.py` | `figure_position_stress_test` |

Run all four with:

```bash
python scripts/generate_figures.py
```

The scripts validate their CSV support and frozen publication values before rendering. Generated PNG/PDF files are ignored by Git; scripts and CSV inputs are tracked.

Appendix Figure B1 is a static model-architecture schematic rather than a data-derived plot, so this repository does not fabricate a plotting script for it.
