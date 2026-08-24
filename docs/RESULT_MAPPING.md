# Result mapping

| Paper concept | Public result | Publication rendering source |
|---|---|---|
| Native counterfactual adequacy | `results/expected/paper_summary.json` → `native_counterfactual_adequacy` | `tables/main_results.md` |
| Low-rank velocity carrier and rank transition | `low_rank_velocity_carrier` and `results/expected/detailed/velocity_carrier_development.json` | `figures/rank_transition/` (Figure 1) |
| Fresh-checkpoint replication and specificity controls | replication fields under `low_rank_velocity_carrier` plus detailed replication/control records | `figures/fresh_replication/` (Figure 2) |
| Temporal per-anchor existence and frozen-anchor transport | `temporal_stability` and `results/expected/detailed/temporal_stability.json` | `figures/temporal_robustness/` (Figure 3) |
| Position-edit negative specificity contrast | `intervention_specificity` and `results/expected/detailed/intervention_specificity.json` | `figures/position_stress_test/` (Figure 4) |
| Corrected Joint behavior and fidelity | `experiments/joint_velocity_composition/data/joint_behavior_summary.csv` | `figures/joint_extension/joint_behavior_panel.csv` |
| Corrected static/dynamic composition | `joint_composition_summary.csv`, `joint_composition_units.csv`, and `joint_dynamic_composition.csv` | `figures/joint_extension/joint_composition_panel.csv` |
| Native vs U4-oracle vs Single-only-addressable Joint behavior | `joint_addressability_summary.csv` and `joint_addressability_units.csv` | `figures/joint_extension/joint_addressability_panel.csv` |
| Corrected specificity controls | `joint_specificity_controls.csv` and `joint_specificity_summary.csv` | public numeric reproduction |
| Rank-4/complement localization and dynamics | `joint_subspace_composition.csv` and `joint_subspace_dynamics.csv` | public mechanistic support |
| Full-patch geometry vs behavior | `joint_patch_geometry.csv` and `joint_patch_geometry_units.csv` | public mechanistic support |

The original figure CSVs beside each plotting script are frozen publication-facing inputs derived from the original detailed results. The corrected Joint figure source tables show checkpoint points as well as family summaries, so checkpoint overlap remains visible.

Run the identity-keyed semantic gate and corrected summary reconstruction with:

```bash
python experiments/joint_velocity_composition/scripts/verify_joint_alignment.py
python experiments/joint_velocity_composition/scripts/reproduce_corrected_joint_results.py
```

Rank 4 denotes the smallest passing tested rank for the original registered single-component grid, not a complete-state dimension. The corrected Joint package is a bounded post-submission extension, and the position result remains a negative contrast.
