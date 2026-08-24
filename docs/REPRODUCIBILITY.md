# Reproducibility modes

## Exact frozen-artifact reproduction

`python scripts/reproduce_main_results.py` is the original paper-facing path. It verifies SHA256 hashes, checks checkpoint topology, exercises numerical components, copies the frozen scientific summary to `results/reproduced/`, and renders the original figures and tables. It does not fit a model or select a rank.

`python scripts/reproduce_all_results.py` adds exhaustive packaged-record validation and loads every checkpoint and fitted carrier. It checks that all JSONL rows parse, arrays are finite, carrier bases are orthonormal within tolerance, and checkpoint parameter counts match the documented topology. It also runs the corrected Joint identity and summary gates.

The corrected Joint extension can be checked independently:

```bash
python experiments/joint_velocity_composition/scripts/verify_joint_alignment.py
python experiments/joint_velocity_composition/scripts/reproduce_corrected_joint_results.py --figure
```

The semantic command validates `joint_request_units.csv`: the exact 512-unit public membership, 411/101 FIT/TEST split, edited object, requested velocity deltas, source-noise identity, and Single-x + Single-y = Joint numerical equality. The join is keyed by explicit public unit identity and is reconstructed in a deliberately different row order.

The summary command recomputes:

- checkpoint and family Joint behavior from `joint_behavior_summary.csv`;
- static composition from 909 unit rows and dynamic composition at the registered horizons;
- Single-only-addressable behavior from 606 unit rows;
- four specificity-control families from 2424 unit rows;
- full-patch geometry from 606 unit rows;
- rank-4/complement family summaries from checkpoint values.

All corrected files, scripts, tests, and figure inputs are bound by `ARTIFACT_MANIFEST.json`. `experiments/joint_velocity_composition/data/SOURCE_MANIFEST.json` gives the narrower public data contract.

## From-scratch training replication

Pass `--retrain` to the full command. Training uses the packaged training split, deterministic seeds, 70 epochs, 24 batches per epoch, batch size 96, AdamW with learning rate 0.0015 and weight decay 0.00001, and cosine decay to 0.00015. Objective weights are fixed in `configs/native_counterfactual_adequacy/training.json`.

The training path writes new checkpoints under `results/reproduced/retraining/`; it never overwrites packaged checkpoints. Exact bytes can vary across devices and library builds, so compare numerical outputs and state hashes rather than archive bytes. Retraining is unnecessary for the corrected Joint release because the correction concerns evaluation source identity, not checkpoint training.

## Determinism

The scripts seed Python, NumPy, and PyTorch. CPU frozen reproduction is the reference. PyTorch deterministic algorithms are requested on CUDA, but platform-dependent numerical differences can remain.

## Scope of generation

The repository redistributes frozen simulator-produced datasets and corrected processed Joint records. It validates their structure and scientific numerical content. Recreating the exact original sampling ledger from an empty directory is not the paper-facing path; packaged hashes and explicit public unit identities are the stable reference.
