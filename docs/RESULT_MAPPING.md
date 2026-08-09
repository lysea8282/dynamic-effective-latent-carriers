# Result mapping

| Paper concept | Public result | Rendering |
|---|---|---|
| Native counterfactual adequacy | `results/expected/paper_summary.json` → `native_counterfactual_adequacy` | `tables/main_results.md` |
| Low-rank velocity carrier | `low_rank_velocity_carrier` | `figures/rank_transition.png` |
| Rank transition | `development_rank_profile` | `figures/rank_transition.png` |
| Addressable vs oracle | `oracle_selected_rank`, `addressable_selected_rank`, replication counts | `tables/main_results.md` |
| Autonomous rollout controls | `mandatory_controls_pass` and detailed velocity records | `tables/main_results.md` |
| Temporal stability | `temporal_stability` | `figures/temporal_stability.png` |
| Event-phase negative result | `event_phase` | `tables/main_results.md` |
| Velocity vs position specificity | `intervention_specificity` | `figures/intervention_specificity.png` |

Files under `results/expected/detailed/` retain the complete numerical result structures with public-facing labels. The concise paper summary is derived from those frozen structures and is the stable rendering interface.
