# Dynamic Effective Latent Carriers

This repository is the reproducibility package for **“Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models.”** It contains the frozen datasets, six trained checkpoints, fitted carrier artifacts, result records, and code needed to inspect or reproduce the reported workflow without another source tree.

## Core result

For the original confirmatory single-component velocity intervention family, rank 4 is the **smallest passing tested rank** on the registered grid. A checkpoint-specific rank-4 dynamics-effective carrier forms a compact, addressable counterfactual intervention interface for the tested velocity-edit regime. This is a one-shot intervention followed by a 12-step autonomous rollout, not a claim that the environment has an intrinsic four-dimensional state.

The confirmatory core also includes fresh-checkpoint replication, temporal reuse, random/wrong-object/wrong-time controls, and a position-edit negative specificity contrast.

## Corrected Joint extension

The post-submission replicated extension evaluates simultaneous two-component velocity requests on the same object. After correcting a source-identity alignment error, the same Single-derived, Single-only-addressed construction—fitted separately for each checkpoint—shows bounded Joint addressability. MF1 already shows strong unseen-Joint competence; MF2 mainly improves rollout fidelity and descriptive target specificity. Corrected native Joint responses become more compositional from MF0 to MF1 to MF2 at the family-median level, with overlap among checkpoint results.

The corrected checkpoint-level and unit-level exports, identity-keyed semantic test, and summary reproduction live in [`experiments/joint_velocity_composition/`](experiments/joint_velocity_composition/). See [`REVISION_NOTE.md`](REVISION_NOTE.md) for the correction scope.

## Mechanistic boundary

The post-submission mechanistic analyses show that composition-relevant error is distributed across the rank-4 carrier and its complement. The carrier is a compact intervention-entry interface; it is **not** established as a closed four-dimensional recurrent state. Geometric closeness to the tested projected native-Joint oracle is not, by itself, a sufficient proxy for dynamics-effective addressability.

## Installation

Python 3.12 is the reference version. A CPU is sufficient for frozen-artifact reproduction; CUDA can accelerate optional retraining.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the repository was obtained through Git with LFS pointer files, materialize the scientific data first:

```bash
git lfs pull
python scripts/verify_artifacts.py
```

## Quick reproduction

```bash
python scripts/reproduce_main_results.py
```

This verifies every manifest-bound artifact, loads a checkpoint, exercises the simulator and rank-4 carrier, writes the frozen paper summary under `results/reproduced/`, regenerates the original publication figures, and regenerates the paper-facing summary tables.

For the corrected Joint extension alone:

```bash
python experiments/joint_velocity_composition/scripts/verify_joint_alignment.py
python experiments/joint_velocity_composition/scripts/reproduce_corrected_joint_results.py --figure
```

The first command validates 512 explicit unit identities, edited objects, velocity deltas, and exact Single-x + Single-y = Joint numerical composition. The second independently recomputes the public checkpoint/family summaries from the released tables.

## Full frozen-artifact validation

```bash
python scripts/reproduce_all_results.py
```

The full path validates every packaged dataset, checkpoint, fitted carrier, detailed result record, and corrected Joint export. It does not repeat expensive scientific model selection. To run a from-scratch training replication as a separate numerical check:

```bash
python scripts/reproduce_all_results.py --retrain --seeds 291101
```

Retraining is substantially slower and may not reproduce byte-identical files across hardware or library builds. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Public experiment readbacks

```bash
python experiments/native_counterfactual_adequacy/run.py
python experiments/low_rank_velocity_carrier/run.py
python experiments/temporal_stability/run.py
python experiments/intervention_specificity/run.py
python experiments/joint_velocity_composition/scripts/reproduce_corrected_joint_results.py
```

These commands read frozen public result records. They are transparent inspection and release tests, not new scientific runs.

## Data and checkpoints

All scientific inputs use repository-relative paths. Hashes, sizes, purposes, and consumers are recorded in `ARTIFACT_MANIFEST.json`, `data/DATASET_MANIFEST.json`, and `checkpoints/CHECKPOINT_MANIFEST.json`.

- `data/` contains packaged training, carrier-fit, development, replication, and temporal records.
- `checkpoints/models/` contains three development and three independent fresh-replication models.
- `checkpoints/carriers/` contains the fitted velocity, temporal, and position carrier bundles used by the public validation path.
- `results/expected/` contains the frozen original-paper summaries.
- `experiments/joint_velocity_composition/` contains the corrected post-submission Joint extension.

See [`docs/DATASETS.md`](docs/DATASETS.md), [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md), and [`docs/RESULT_MAPPING.md`](docs/RESULT_MAPPING.md).

## Figures

```bash
python scripts/generate_figures.py
```

This regenerates the four original publication figures plus a public draft of the corrected Joint-extension figure. Each plotting directory contains its source CSV data and script; generated PNG/PDF files are ignored by Git. See [`figures/README.md`](figures/README.md).

## Scope

All findings are bounded to the packaged deterministic simulator, model family, edit families, checkpoints, horizons, and thresholds. Same procedure across checkpoints does not mean the checkpoints share one raw hidden basis. The Joint extension is limited to same-object two-component velocity requests and does not establish a universal intervention algebra or causal law of training. Position editing remains a negative contrast. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Citation and license

Citation metadata is in `CITATION.cff`. Code and packaged research artifacts are distributed under the included MIT License; users remain responsible for checking requirements that apply to redistributed or derived bundles.

## AI Assistance Disclosure

ChatGPT and OpenAI Codex were used to assist with literature organization, experimental and code development, analysis organization, and manuscript editing. All scientific decisions, experimental results, interpretations, and final claims were independently reviewed and determined by the authors, who take full responsibility for the content of this work.
