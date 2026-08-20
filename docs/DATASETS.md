# Datasets

`data/DATASET_MANIFEST.json` is the machine-readable inventory. It records repository-relative paths, SHA256 hashes, sizes, row counts or array shapes, and consuming experiments.

- `data/training/model_training_dataset.npz` contains factual and counterfactual trajectories, observations, actions, split labels, and neutral public metadata.
- `data/training/velocity_carrier_fit.jsonl` supplies the velocity carrier fit set.
- `data/evaluation/velocity_*.jsonl` supplies development and replication evaluations.
- `data/counterfactual/temporal_*.jsonl` supplies temporal carrier fitting, development, and replication panels.

All numerical arrays and JSON numeric values are preserved from the frozen scientific artifacts. Text identifiers were replaced with neutral public identifiers; the manifest records numerical-content hashes to verify this transformation.
