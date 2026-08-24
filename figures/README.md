# Paper figure reproduction

Each subdirectory contains a self-contained plotting script and public CSV inputs. Scripts use paths relative to their directory and write PNG and PDF outputs beside the script.

| Figure | Public source directory | Command from repository root | Generated basename |
|---|---|---|---|
| Original Figure 1 — rank transition and carrier spectrum | `figures/rank_transition/` | `python figures/rank_transition/plot_rank_transition_and_spectrum.py` | `figure_rank_transition_and_spectrum` |
| Original Figure 2 — fresh replication and controls | `figures/fresh_replication/` | `python figures/fresh_replication/plot_fresh_replication_and_controls.py` | `figure_fresh_replication_and_controls` |
| Original Figure 3 — temporal robustness | `figures/temporal_robustness/` | `python figures/temporal_robustness/plot_figure_temporal_robustness.py` | `figure_temporal_robustness` |
| Original Figure 4 — position-edit stress test | `figures/position_stress_test/` | `python figures/position_stress_test/plot_position_stress_test.py` | `figure_position_stress_test` |
| Corrected Joint extension (public draft) | `figures/joint_extension/` | `python figures/joint_extension/make_joint_extension_figure.py` | `figure_joint_extension` |

Run all five with:

```bash
python scripts/generate_figures.py
```

The corrected Joint draft has three panels: corrected behavior/fidelity, static composition, and Native vs U4-oracle vs Single-only-addressable coverage. Checkpoint points are shown rather than only family bars. Generated PNG/PDF files are ignored by Git; scripts and CSV inputs are tracked and hash-bound.

Appendix Figure B1 is a static model-architecture schematic rather than a data-derived plot, so this repository does not fabricate a plotting script for it.
