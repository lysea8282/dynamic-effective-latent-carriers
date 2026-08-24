# Corrected Joint velocity extension

This package releases the corrected post-submission analysis of same-object, simultaneous two-component velocity requests. It is a bounded extension of the original single-component result, using the same Single-derived, Single-only-addressed construction fitted separately for each checkpoint. No Joint-specific addressability mapper is fitted.

## Reproduce and verify

Run from the repository root:

```bash
python experiments/joint_velocity_composition/scripts/verify_joint_alignment.py
python experiments/joint_velocity_composition/scripts/reproduce_corrected_joint_results.py --figure
python -m pytest experiments/joint_velocity_composition/tests -q
```

The alignment command performs a scientific semantic test over 512 canonical units. It joins by explicit public unit identity and verifies membership, edited object, velocity deltas, source-noise identity, no duplicates/missing units, and exact Single-x + Single-y = Joint composition. It intentionally reconstructs records in a different row order to rule out positional binding.

The reproduction command recomputes checkpoint and family values from the public CSVs, including 909 static-composition unit rows, 606 addressability rows, 2424 specificity controls, and 606 full-patch geometry rows, then renders the selected three-panel public figure.

## Public data

| File | Role |
|---|---|
| `data/joint_request_units.csv` | 512-unit identity and numerical-semantic fixture |
| `data/joint_behavior_summary.csv` | corrected M0/M1/M2 coverage and anchor/k1/t19 fidelity |
| `data/joint_composition_units.csv` | unit-level static composition descriptors |
| `data/joint_composition_summary.csv` | checkpoint and family static composition |
| `data/joint_dynamic_composition.csv` | checkpoint-level dynamic composition at registered horizons |
| `data/joint_addressability_units.csv` | same-101 Single-only-addressable results |
| `data/joint_addressability_summary.csv` | Native vs U4-oracle vs Single-only-addressable comparison |
| `data/joint_specificity_controls.csv` | 2424 wrong-object, wrong-vector, random, and sham unit controls |
| `data/joint_specificity_summary.csv` | checkpoint-level control separations |
| `data/joint_subspace_composition.csv` | rank-4/complement composition decomposition |
| `data/joint_subspace_dynamics.csv` | rank-4/complement dynamic error descriptors |
| `data/joint_patch_geometry_units.csv` | unit-level full-patch geometry |
| `data/joint_patch_geometry.csv` | geometry with matched behavioral context |
| `expected/corrected_joint_summary.json` | frozen public headline summary |

`data/SOURCE_MANIFEST.json` binds each export by SHA256, row count, identity contract, and scientific role. The repository-level `ARTIFACT_MANIFEST.json` additionally binds the complete package, scripts, tests, and figure-source data.

## Claim boundary

- Rank 4 remains the smallest passing tested rank for the original single-component registered grid; it is not an intrinsic-state dimension claim.
- The corrected Joint extension is limited to same-object two-component velocity requests and checkpoint-specific fits.
- The same procedure across checkpoints does not mean the raw hidden bases are shared.
- M1 already has strong unseen-Joint competence; M2 mainly improves rollout fidelity and descriptive specificity.
- Composition improves at the family level with checkpoint overlap and is distributed across the rank-4 carrier and its complement.
- A compact intervention-entry carrier is not a closed four-dimensional recurrent trajectory model.
- Position editing remains a negative specificity contrast.

These are post-submission replicated and mechanistic analyses, not preregistered confirmatory results.
