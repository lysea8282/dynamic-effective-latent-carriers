# Dynamic Effective Latent Carriers

This repository is the reproducibility package for the paper **“Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models.”** It contains the frozen datasets, six trained model checkpoints, fitted carrier artifacts, result records, and code needed to inspect or reproduce the reported workflow without another source tree.

The scientific scope is deliberately narrow: native counterfactual adequacy; a tested low-rank velocity carrier; oracle and addressable edits; autonomous-rollout controls; independent fresh-checkpoint replication; temporal per-anchor existence and frozen-anchor transport; a bounded event-phase result; and a velocity-versus-position specificity diagnostic.

## Installation

Python 3.12 is the reference version. A CPU is sufficient for frozen-artifact reproduction; CUDA can accelerate optional retraining.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the repository was obtained through Git with LFS pointer files, materialize the packaged scientific data first:

```bash
git lfs pull
python scripts/verify_artifacts.py
```

## Quick reproduction

```bash
python scripts/reproduce_main_results.py
```

This verifies every manifest-bound artifact, loads a checkpoint, exercises the simulator and rank-4 carrier, writes the frozen paper summary under `results/reproduced/`, regenerates publication Figures 1–4, and regenerates the paper-facing summary tables. On a typical CPU it takes about one minute, dominated by hashing.

## Full frozen-artifact validation

```bash
python scripts/reproduce_all_results.py
```

The full path validates every packaged dataset, checkpoint, fitted carrier, and detailed result record. It does not repeat expensive scientific model selection. To run a from-scratch model-training replication as a separate numerical check:

```bash
python scripts/reproduce_all_results.py --retrain --seeds 291101
```

Retraining is substantially slower and may not reproduce byte-identical files across hardware or library builds. The frozen path is the exact paper-artifact reproduction; retraining is a numerical replication of the documented objective. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Public experiment readbacks

The following repository-relative commands expose the packaged result structures for the principal experiment families:

```bash
python experiments/native_counterfactual_adequacy/run.py
python experiments/low_rank_velocity_carrier/run.py
python experiments/temporal_stability/run.py
python experiments/intervention_specificity/run.py
```

These commands read the frozen public result records. They are intended for transparent inspection and release testing, not as new scientific runs.

## Data and checkpoints

All scientific inputs use repository-relative paths. Hashes, sizes, purposes, and consumers are recorded in `ARTIFACT_MANIFEST.json`, `data/DATASET_MANIFEST.json`, and `checkpoints/CHECKPOINT_MANIFEST.json`.

- `data/` contains the packaged training, carrier-fit, development, replication, temporal, and event-phase records.
- `checkpoints/models/` contains three development and three independent fresh-replication models.
- `checkpoints/carriers/` contains the fitted velocity, temporal, position, and event-phase carrier bundles used by the public validation path.
- `results/expected/` contains the frozen paper summaries and detailed result structures.

See [docs/DATASETS.md](docs/DATASETS.md) and [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Reproduce paper Figures 1–4

Run all four publication-layout figure scripts with:

```bash
python scripts/generate_figures.py
```

Or run each figure directly from the repository root:

```bash
python figures/rank_transition/plot_rank_transition_and_spectrum.py
python figures/fresh_replication/plot_fresh_replication_and_controls.py
python figures/temporal_robustness/plot_figure_temporal_robustness.py
python figures/position_stress_test/plot_position_stress_test.py
```

Each script reads only the CSV file or files in its own directory and writes a PNG and PDF beside the script. Generated images are intentionally ignored by Git; the plotting scripts and source CSV files are tracked. Exact inputs and outputs are listed in [figures/README.md](figures/README.md).

## Expected paper-scope results

- All six native-model seeds satisfy the packaged adequacy panels.
- Rank 4 is the **smallest passing tested intervention rank** for both oracle and addressable velocity edits on the registered grid. It is not a claim that the environment has a complete four-dimensional state or that four dimensions are universally sufficient.
- Two of three fresh replication checkpoints satisfy both registered strata.
- The tested carrier succeeds at all five per-anchor existence panels and all four frozen-`t=7` transport targets under the registered model-level requirement.
- The bounded event-phase development assay provides no positive signal.
- Position editing is a **negative specificity contrast**, not a second positive carrier: raw success is also achieved by negative-control routes, while the registered specificity conditions fail.

The exact claim-to-file mapping is in [docs/RESULT_MAPPING.md](docs/RESULT_MAPPING.md). These findings are bounded to the packaged simulator, model family, edit families, seeds, horizons, and thresholds; see [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Outputs

- Frozen reproduction records: `results/reproduced/`
- Publication Figures 1–4: the corresponding `figures/<figure-name>/` directories
- Summary tables: `tables/`

Generated outputs can be deleted and regenerated; packaged data, checkpoints, source CSV files, and expected result records are the stable inputs.

## Hardware and runtime

- Frozen quick path: CPU, about one minute, less than 4 GB RAM.
- Frozen full validation: CPU, about 2–5 minutes, less than 8 GB RAM.
- One training seed: CPU hours or a modern GPU; six seeds require proportionally more time.
- Reference package versions: Python 3.12.10, PyTorch 2.11.0, NumPy 2.4.4.

## Citation and license

Citation metadata is in `CITATION.cff`. Code and packaged research artifacts are distributed under the included MIT License; users remain responsible for checking requirements that apply to redistributed or derived bundles.

## AI Assistance Disclosure

ChatGPT and OpenAI Codex were used to assist with literature organization, experimental and code development, analysis organization, and manuscript editing. All scientific decisions, experimental results, interpretations, and final claims were independently reviewed and determined by the authors, who take full responsibility for the content of this work.

