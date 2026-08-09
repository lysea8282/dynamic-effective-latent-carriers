# Dynamic Effective Latent Carriers

This repository reproduces the experiments accompanying the paper on low-rank, dynamics-effective latent carriers in a deterministic two-object collision world. It contains the frozen datasets, six trained model checkpoints, fitted carrier artifacts, result records, and command-line rendering code needed to check the reported results without another source tree.

The scientific scope is deliberately narrow: native counterfactual adequacy, a rank-4 velocity carrier, addressable versus oracle edits, autonomous-rollout controls, temporal stability, an event-phase negative result, and a velocity-versus-position specificity diagnostic.

## Installation

Python 3.12 is the reference version. A CPU is enough for frozen-artifact reproduction; CUDA can accelerate optional retraining.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick reproduction

```bash
python scripts/reproduce_main_results.py
```

This verifies every packaged artifact, loads a checkpoint, exercises the simulator and rank-4 carrier, materializes the frozen paper summary under `results/reproduced/`, and regenerates the paper-facing figures and tables. On a typical CPU it takes about one minute, dominated by hashing.

## Full reproduction

```bash
python scripts/reproduce_all_results.py
```

The default full path verifies all datasets, checkpoints, fit bundles, and detailed result records. It performs bounded numerical checks over every packaged record and fit artifact. To run a from-scratch model-training replication as well:

```bash
python scripts/reproduce_all_results.py --retrain --seeds 291101
```

Retraining is substantially slower and may not reproduce byte-identical files across hardware or library builds. The frozen path is the exact paper-artifact reproduction; retraining is a numerical replication of the documented objective. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Data and checkpoints

All scientific inputs are repository-relative. Hashes, sizes, purposes, and consumers are recorded in `ARTIFACT_MANIFEST.json`, `data/DATASET_MANIFEST.json`, and `checkpoints/CHECKPOINT_MANIFEST.json`. Large binary and line-delimited data files are tracked with Git LFS. Run:

```bash
git lfs pull
python scripts/verify_artifacts.py
```

See [docs/DATASETS.md](docs/DATASETS.md) and [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Figures and tables

```bash
python scripts/generate_figures.py
python scripts/generate_tables.py
```

These commands consume only `results/expected/` or `results/reproduced/` and write deterministic filenames to `figures/` and `tables/`. Scientific computation remains separate from paper rendering.

## Expected results

The main findings are: all six native-model seeds pass their prespecified adequacy panels; the smallest passing development rank is 4 for both oracle and addressable velocity edits; the frozen replication panel passes for two of three checkpoints; the carrier remains successful at all five tested anchors and all four tested transport targets; event-phase analyses provide no positive signal in the bounded development test; and position edits fail specificity controls at every tested amplitude. The exact claim-to-file mapping is in [docs/RESULT_MAPPING.md](docs/RESULT_MAPPING.md).

## Hardware and runtime

- Frozen quick path: CPU, about 1 minute, less than 4 GB RAM.
- Frozen full validation: CPU, about 2–5 minutes, less than 8 GB RAM.
- One training seed: CPU hours or a modern GPU; six seeds require proportionally more time.
- Reference package versions: Python 3.12.10, PyTorch 2.11.0, NumPy 2.4.4.

## Citation and license

Citation metadata is in `CITATION.cff`. Code and packaged research artifacts are released under the MIT License; check your local requirements before redistributing derived bundles.
