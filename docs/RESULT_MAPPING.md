# Result mapping

| Paper concept | Public result | Publication rendering source |
|---|---|---|
| Native counterfactual adequacy | `results/expected/paper_summary.json` → `native_counterfactual_adequacy` | `tables/main_results.md` |
| Low-rank velocity carrier and rank transition | `low_rank_velocity_carrier` and `results/expected/detailed/velocity_carrier_development.json` | `figures/rank_transition/` (Figure 1) |
| Fresh-checkpoint replication and specificity controls | replication fields under `low_rank_velocity_carrier` plus the detailed replication/control records | `figures/fresh_replication/` (Figure 2) |
| Temporal B1 per-anchor existence and B2 frozen-anchor transport | `temporal_stability` and `results/expected/detailed/temporal_stability.json` | `figures/temporal_robustness/` (Figure 3) |
| Position-edit negative specificity contrast | `intervention_specificity` and `results/expected/detailed/intervention_specificity.json` | `figures/position_stress_test/` (Figure 4) |
| Event-phase bounded negative result | `event_phase` | `tables/main_results.md` |

The CSV files beside each public plotting script are the publication-facing rendering inputs. They are derived from the frozen detailed results and include internal consistency checks in the plotting scripts. The scripts do not recompute or reinterpret the scientific results.

Files under `results/expected/detailed/` retain the complete numerical result structures with public-facing labels. The concise paper summary is the stable interface used by the frozen reproduction and summary-table paths.

Rank 4 denotes the smallest passing tested intervention rank on the registered grid; it is not a complete-state dimensionality claim. The position-edit result is a negative specificity contrast, not evidence for a second successful carrier.
