# Reproducibility modes

## Exact frozen-artifact reproduction

`python scripts/reproduce_main_results.py` is the paper-facing path. It verifies SHA256 hashes, checks the packaged checkpoint topology, exercises numerical components, copies the frozen scientific summary to `results/reproduced/`, and renders figures and tables. This path does not fit a new model or select a new rank.

`python scripts/reproduce_all_results.py` adds exhaustive packaged-record validation and loads every checkpoint and fitted carrier. It checks that all JSONL rows parse, all arrays are finite, carrier bases are orthonormal within tolerance, and checkpoint parameter counts match the documented topology.

The full command also runs the lightweight joint-velocity boundary evidence replay. Its component commands live under `experiments/joint_velocity_composition/scripts/`; they validate compact public tables, reproduce the joint-capacity, specificity-calibration, layer-localization, and operator-extension summaries, and compare each summary with its frozen expected record.

## From-scratch training replication

Pass `--retrain` to the full command. Training uses the packaged training split, deterministic seeds, 70 epochs, 24 batches per epoch, batch size 96, AdamW with learning rate 0.0015 and weight decay 0.00001, and cosine decay to 0.00015. The objective weights are fixed in `configs/native_counterfactual_adequacy/training.json`.

The training path writes new checkpoints under `results/reproduced/retraining/`; it never overwrites packaged checkpoints. Exact bytes can vary across devices and library builds, so compare numerical outputs and state hashes rather than archive bytes.

## Determinism

The scripts seed Python, NumPy, and PyTorch. CPU frozen reproduction is the reference. When running on CUDA, PyTorch deterministic algorithms are requested, but some platform-dependent numerical differences can remain.

## Scope of generation

The repository redistributes the frozen simulator-produced datasets. It validates their structure and scientific numerical content. Recreating the exact dataset selection from an empty directory is not the paper-facing path because it would require reproducing the original sampling ledger; the packaged hashes are the stable reference.
