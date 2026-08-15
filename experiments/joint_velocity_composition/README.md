# Joint-velocity composition boundary

## Scientific question

This development-only analysis asks whether the rank-4 carrier established for bounded single-component velocity edits also supports simultaneous two-component velocity changes applied to the same object.

## Primary observation

Native joint edits and their rank-4 oracle projections retain the tested joint-rollout capacity. For the original affine interface, direct joint coefficients and component-wise affine composition agree to numerical precision, and absolute addressable joint rollouts can pass.

## Boundary

Target specificity is weaker than in the established single-component result, especially in interaction-sensitive `S2` rollouts. Retrospective calibration shows that the same 0.80 paired-dominance comparator is compatible with the known-positive single-component baseline. Layer localization retains several distinctions at the native and rank-4 oracle layers but loses them at the addressable layer; wrong-object specificity also has a native ceiling in part of `S2`. Neither a joint-aware affine refit nor a minimal same-object interaction correction closes the full specificity gap.

The interpretation is:

```text
carrier capacity != compositional addressability
```

This is a boundary result, not a second positive intervention family. It does not change the primary single-component confirmatory claim, establish a universal velocity coordinate system, show that rank 4 lacks joint information, or establish full joint compositional control. The analysis uses development models only and is not a fresh-checkpoint confirmation.

## Reproduction

Run from the repository root:

```bash
python experiments/joint_velocity_composition/scripts/reproduce_joint_composition.py
python experiments/joint_velocity_composition/scripts/reproduce_specificity_calibration.py
python experiments/joint_velocity_composition/scripts/reproduce_specificity_layer_localization.py
python experiments/joint_velocity_composition/scripts/reproduce_operator_extension.py
python experiments/joint_velocity_composition/scripts/summarize_joint_composition_boundary.py
```

Each command verifies the compact public source manifest, recomputes the corresponding summary, checks it against `expected/`, and writes a copy under `results/reproduced/joint_velocity_composition/`. The final command reproduces all five summaries.

## Public evidence

- `data/joint_composition_units.csv` contains the 512 bounded joint-edit units with public identifiers.
- `data/joint_composition_results.csv` contains per-model, per-unit native, oracle, affine, and control metrics.
- `data/joint_specificity_pairs.csv` and `data/affine_composition_equivalence.csv` support the paired-specificity and algebraic-equivalence checks.
- `data/single_component_specificity_baseline.csv` supports the retrospective comparator calibration.
- `data/specificity_layer_results.csv` and `data/specificity_layer_localization.csv` support native-to-oracle-to-addressable localization.
- The operator-extension tables and `data/operator_weights.json` support the refit, interaction-correction, and backward-compatibility checks.
- `data/public_source_manifest.json` records public labels, counts, frozen settings, thresholds, hashes, and code identities.

Large observation trajectories, hidden-state payloads, private review archives, and exploratory predecessor records are intentionally omitted because the compact records contain every value consumed by the public summaries and tests.
