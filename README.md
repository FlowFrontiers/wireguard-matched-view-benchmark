# WireGuard Matched-View Benchmark

Reproducible Python pipeline for building the matched-view benchmark, running
its frozen experiment matrices, and producing the machine-readable results used
by the accompanying paper. The code operates on paired observations of the
same physical flows before and after WireGuard encapsulation.

This repository contains the journal experiment pipeline. It is separate from
the frozen conference-paper repository.

## What This Repository Runs

The pipeline provides:

- deterministic construction of canonical inner/outer flow pairs from the
  four published dataset artifacts;
- pair-disjoint five-fold evaluation across inner and outer views;
- bidirectional cross-session evaluation;
- Random Forest, XGBoost, CNN1D, LSTM, and Transformer baselines;
- CNN1D-based domain-adversarial training (DANN);
- packet-prefix and SPLT-channel ablations;
- content-validated predictions, metrics, seed ensembles, and paired bootstrap
  confidence intervals.

The frozen campaign contains 265 unique model trainings after excluding reused
ablation reference cells. Neural model selection adds 36 development-fold
trials: 12 each for CNN1D, LSTM, and Transformer.

## Repository Layout

```text
configs/       Frozen dataset, feature, model, protocol, and analysis settings
scripts/       Script equivalents of the installed vpncat-* commands
src/vpncat/    Dataset, preprocessing, training, orchestration, and analysis code
tests/         Synthetic unit, contract, leakage, and artifact-integrity tests
data/raw/      Four downloaded dataset inputs (not tracked)
data/processed Deterministic canonical data and contract audits (not tracked)
outputs/       Tuning, run, report, and aggregate artifacts (not tracked)
docs/          Detailed protocol and CLI reference
```

## Requirements

- Python 3.11 or newer; Python 3.12 is recommended.
- XGBoost for the complete classical matrix.
- PyTorch for neural tuning and execution.
- An NVIDIA CUDA GPU is recommended for the full campaign. Apple MPS and CPU
  execution are supported for development and bounded smoke tests.

Create the environment and install all pipeline components:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[classical,neural,dev]'
```

Verify the installation without downloading the dataset:

```bash
pytest
ruff check .
```

The tests use synthetic paired data. They do not train the full experiment
matrix.

## Reproducibility Rule

Run the campaign from a clean checkout of the paper's tagged release. Do not
edit or commit files after tuning or experiment execution begins. Every
contract, tuning selection, and published run records the Git revision and
input hashes; stale or dirty state is rejected rather than silently reused.

Neural tuning and matrix execution are resumable. Run the build, audit, and
contract commands once in the documented order; their overwrite behavior is
command-specific. Final aggregation refuses an existing analysis directory.
Existing run artifacts are accepted only after their identities, hashes,
predictions, and recomputed metrics validate against the current clean
revision.

## Dataset

Download version 3.0.0 of the public
[VPN-nonVPN-Dataset](https://doi.org/10.5281/zenodo.18945858). No PCAP files are
required. Place these four Parquet files as follows:

```text
data/raw/
├── session1/
│   ├── session1_flows.parquet
│   └── session1_packet_matches.parquet
└── session2/
    ├── session2_flows.parquet
    └── session2_packet_matches.parquet
```

The pipeline uses the released flow tables and packet-match tables. It does not
use legacy CSV exports or source-derived aggregate columns. Alternative input
and output locations can be supplied through CLI path overrides; run
`vpncat-build-dataset --help` for details.

## Prepare the Benchmark

Build and validate the canonical pair table and fixed five-fold split:

```bash
vpncat-build-dataset --config configs/dataset.yaml
vpncat-validate-dataset --config configs/dataset.yaml
```

The builder reproduces packet-to-flow assignment, checks every released matched
packet, and recomputes both views under one statistical convention. It writes
the canonical data, split, assignment audit, and content manifest under
`data/processed/`.

Materialize the feature, preprocessing, and experiment contracts in this order:

```bash
vpncat-audit-features --config configs/features.yaml
vpncat-audit-preprocessing --config configs/preprocessing.yaml
vpncat-audit-experiment-contract --config configs/primary.yaml

vpncat-cross-session-contract --config configs/cross_session.yaml
vpncat-audit-cross-session-preprocessing \
  --config configs/cross_session_preprocessing.yaml

vpncat-dann-contract --config configs/dann.yaml
vpncat-ablation-contract --config configs/ablation_prefix.yaml
vpncat-ablation-contract --config configs/ablation_channels.yaml
vpncat-analysis-contract --config configs/analysis.yaml
```

These commands fail on inconsistent pair membership, leakage-prone fitted
state, contract drift, or stale upstream artifacts.

## Tune Neural Models

Tuning is performed once on fold 1 using only the inner training and validation
views. Held-out pairs and outer-view features are not materialized during model
selection. Run all 12 frozen trials for each architecture in one consistent
software environment and on the same device type. The full primary controller
rejects selections produced by different tuning environments:

```bash
vpncat-tune-neural --model cnn1d --device cuda
vpncat-tune-neural --model lstm --device cuda
vpncat-tune-neural --model transformer --device cuda
```

Final runs may execute in another environment; each run records its own
software versions and device metadata.

Rerunning a command validates and reuses completed trials. Individual trials can
be resumed with repeatable `--trial-id` options. A selection is published only
when all 12 trials for that architecture are complete and valid.

For Apple Silicon smoke testing, replace `cuda` with `mps`. Use `cpu` only when
GPU execution is unavailable.

## Preflight the Campaign

Controllers run in plan-only mode unless `--execute` is supplied. Preflight all
protocols before starting a long campaign:

```bash
vpncat-primary-matrix --report outputs/reports/primary-plan.json
vpncat-cross-session-matrix --report outputs/reports/cross-session-plan.json
vpncat-dann-matrix --config configs/dann.yaml \
  --report outputs/reports/dann-plan.json
vpncat-ablation-matrix --config configs/ablation_prefix.yaml \
  --report outputs/reports/ablation-prefix-plan.json
vpncat-ablation-matrix --config configs/ablation_channels.yaml \
  --report outputs/reports/ablation-channels-plan.json
```

A fresh campaign must report:

| Protocol | Executable trainings | Reused references |
|---|---:|---:|
| Primary matrix | 150 | 0 |
| Cross-session matrix | 30 | 0 |
| DANN matrix | 15 | 0 |
| Prefix ablation | 30 | 10 |
| Channel ablation | 40 | 10 |
| **Total** | **265** | **20 logical references** |

The 20 logical ablation references resolve to 10 unique seed-42 primary
artifacts. They are never retrained by the ablation controllers.

## Smoke Test

After tuning, execute and validate one representative neural primary run before
starting the full matrix:

```bash
vpncat-primary-matrix \
  --model cnn1d \
  --fold 1 \
  --train-domain inner \
  --seed 42 \
  --execute \
  --maximum-pending-runs 1 \
  --device cuda \
  --report outputs/reports/primary-smoke.json
```

The later full primary command detects this valid run and resumes with the
remaining entries.

## Run the Full Campaign

Run the controllers in the following order. Each command performs a complete
preflight before training and revalidates every newly published run before
continuing.

```bash
vpncat-primary-matrix \
  --execute \
  --device cuda \
  --report outputs/reports/primary-final.json

vpncat-cross-session-matrix \
  --execute \
  --device cuda \
  --report outputs/reports/cross-session-final.json

vpncat-dann-matrix \
  --config configs/dann.yaml \
  --execute \
  --device cuda \
  --report outputs/reports/dann-final.json

vpncat-ablation-matrix \
  --config configs/ablation_prefix.yaml \
  --execute \
  --device cuda \
  --report outputs/reports/ablation-prefix-final.json

vpncat-ablation-matrix \
  --config configs/ablation_channels.yaml \
  --execute \
  --device cuda \
  --report outputs/reports/ablation-channels-final.json
```

Rerun the same command after an interruption. Valid completed runs are checked
and skipped. A partial, stale, or incompatible run directory stops the campaign
with a diagnostic instead of being overwritten.

Use `--maximum-pending-runs 1` to execute one pending run at a time. Controllers
also provide filters for model, representation, fold/session, seed, and exact
run ID; inspect each command with `--help` before partitioning work across
machines. Do not point overlapping concurrent controllers at the same output
root.

## Aggregate Results

When every controller reports complete, publish the final analysis bundle:

```bash
vpncat-aggregate-results --config configs/analysis.yaml
```

Aggregation validates all 265 physical artifacts and their upstream contracts
before reading predictions. It averages neural seeds at the class-probability
level, retains per-seed dispersion, and computes paired 1,000-replicate
pair-level bootstrap intervals for balanced accuracy, macro F1, and the
outer-minus-inner gap.

The command writes `outputs/analysis/` atomically:

```text
outputs/analysis/
├── analysis.json
├── logical_aliases.csv
├── metrics_summary.csv
├── seed_metrics.csv
├── seed_dispersion.csv
├── bootstrap_intervals.csv
└── predictions/
    └── <physical-group-id>.parquet
```

The CSV files are the machine-readable sources for paper tables and figures.
Aggregation refuses to overwrite an existing analysis directory.

## Run Artifacts

Each trained model publishes one immutable directory containing both inner and
outer held-out predictions:

```text
run.json
split_manifest.csv
metrics.json
metrics_long.csv
predictions.parquet
training_history.csv  # all neural runs, including DANN and ablations
```

`run.json` binds the model identity, configuration hashes, fitted-state hashes,
Git revision, package versions, execution environment, neural device where
applicable, and input pair set. The prediction file includes pair identity,
session or fold context, training and test domains, true and predicted labels,
and class probabilities.

Default output roots are:

```text
outputs/tuning/
outputs/primary/
outputs/cross_session/
outputs/dann/
outputs/ablations/prefix/
outputs/ablations/channels/
outputs/analysis/
```

## Validation and Recovery

- Run any matrix controller without `--execute` to validate its completed
  outputs and report the remaining inventory.
- Rerun interrupted tuning or matrix commands; valid atomic outputs are reused.
- Never use `--force` during a campaign. It is reserved for deliberate contract
  regeneration before training starts.
- Do not manually repair hashes or edit generated JSON/CSV/Parquet files.
  Validators recompute metrics and content invariants, not only checksums.
- XGBoost is automatically isolated in a subprocess to avoid PyTorch/OpenMP
  runtime collisions on macOS.
- Keep `data/processed/` and `outputs/` on persistent storage during cloud
  execution.

## Detailed Reference

See [docs/protocol-reference.md](docs/protocol-reference.md) for feature
semantics, leakage controls, model contracts, individual-run commands, artifact
validation rules, and protocol-specific implementation details.

All installed commands have equivalent wrappers under `scripts/` and provide
complete option documentation through `--help`.

## Citation

When publishing results produced by this code, cite the accompanying paper and
the source dataset:

> Razooqi, Y. S., and Pekar, A. (2026). VPN-nonVPN-Dataset (v3.0.0) [Data set].
> Zenodo. https://doi.org/10.5281/zenodo.18945858

## License

MIT. See [LICENSE](LICENSE).
