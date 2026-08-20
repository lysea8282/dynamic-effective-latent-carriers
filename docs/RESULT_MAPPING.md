# Result mapping

| Paper concept | Public result | Publication rendering source |
|---|---|---|
| Native counterfactual adequacy | `results/expected/paper_summary.json` → `native_counterfactual_adequacy` | `tables/main_results.md` |
| Low-rank velocity carrier and rank transition | `low_rank_velocity_carrier` and `results/expected/detailed/velocity_carrier_development.json` | `figures/rank_transition/` (Figure 1) |
| Fresh-checkpoint replication and specificity controls | replication fields under `low_rank_velocity_carrier` plus the detailed replication/control records | `figures/fresh_replication/` (Figure 2) |
| Temporal B1 per-anchor existence and B2 frozen-anchor transport | `temporal_stability` and `results/expected/detailed/temporal_stability.json` | `figures/temporal_robustness/` (Figure 3) |
| Position-edit negative specificity contrast | `intervention_specificity` and `results/expected/detailed/intervention_specificity.json` | `figures/position_stress_test/` (Figure 4) |
| Development-only joint-velocity composition boundary | `experiments/joint_velocity_composition/expected/boundary_claim_summary.json` | No manuscript figure; compact public evidence replay |

The CSV files beside each public plotting script are the publication-facing rendering inputs. They are derived from the frozen detailed results and include internal consistency checks in the plotting scripts. The scripts do not recompute or reinterpret the scientific results.

Files under `results/expected/detailed/` retain the complete numerical result structures with public-facing labels. The concise paper summary is the stable interface used by the frozen reproduction and summary-table paths.

Rank 4 denotes the smallest passing tested intervention rank on the registered grid; it is not a complete-state dimensionality claim. The position-edit result is a negative specificity contrast, not evidence for a second successful carrier.

The primary confirmed intervention family concerns bounded single-component velocity edits. The joint-velocity package is a separate development-only boundary test: retained carrier capacity does not by itself establish compositional addressability or full target-specific joint control.
