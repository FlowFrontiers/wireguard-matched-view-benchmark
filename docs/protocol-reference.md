# Protocol and CLI Reference

Detailed implementation contracts and individual command reference for the WireGuard Matched-View Benchmark. For the ordered reproduction workflow, start with the repository [README](../README.md).

## Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Install the optional XGBoost dependency on machines that will execute those
runs:

```bash
python -m pip install -e '.[classical,dev]'
```

Install PyTorch for neural contract tests and subsequent GPU execution:

```bash
python -m pip install -e '.[neural,dev]'
```

## Input Data

Obtain the two flow and two packet-match files from the published matched
WireGuard dataset and place them under the configured input root:

```text
data/raw/
├── session1/
│   ├── session1_flows.parquet
│   └── session1_packet_matches.parquet
└── session2/
    ├── session2_flows.parquet
    └── session2_packet_matches.parquet
```

These are public Zenodo artifacts from the DiB dataset; no PCAPs are required.
The builder reproduces assignment from packet 5-tuples and flow time windows,
derives initiator-relative direction from each flow's source and destination
endpoints, then computes both views from the same assigned physical packet
pairs. The packet-match file's capture-relative direction field, legacy CSV
exports, and source-derived aggregate columns are not used.

Paths are configured in `configs/dataset.yaml`. They may be overridden without
source edits, including on Windows:

```bash
vpncat-build-dataset \
  --config configs/dataset.yaml \
  --input-root /path/to/raw-data \
  --output-dir /path/to/processed-data
```

Use `--force` only when intentionally replacing an existing canonical build.

## Generated Artifacts

```text
data/processed/
├── canonical_pairs.parquet
├── split_manifest.csv
├── cross_session_split_manifest.csv
├── assignment_audit.parquet
├── feature_audit.json
├── preprocessing_audit.json
├── experiment_contract_audit.json
├── cross_session_contract_audit.json
├── cross_session_preprocessing_audit.json
├── dann_contract_audit.json
├── ablation_prefix_contract_audit.json
├── ablation_channels_contract_audit.json
├── analysis_contract_audit.json
└── dataset_manifest.json
```

The canonical Parquet file has one row per retained physical flow. MatchedFlowStats
uses every matched packet pair in each flow. Prefix membership is selected by
inner/source order; each view then orders the identical selected packet set by its
own observed timestamp and packet index. PIAT uses consecutive gaps, standard
deviations use `ddof=1`, and rates are missing when duration is not positive.
Absolute capture timestamps are never written.

`assignment_audit.parquet` records released and reproduced counts for all source
flows, including zero-match flows, plus directional-byte fidelity for fully
matched flows. The manifest hashes all four inputs and three generated artifacts,
and records assignment, endpoint-orientation, eligibility, class, and cross-view
reordering diagnostics.

Validate an existing build with:

```bash
vpncat-validate-dataset --config configs/dataset.yaml
```

Validation recomputes every artifact digest and rejects any per-flow assignment
count mismatch. It also checks endpoint-orientation and directional-byte fidelity,
all 42 statistics, rate equations, paired sequence lengths, matched packet counts,
direction and magnitude domains, and exact PrefixStats agreement for flows fully
represented within the stored prefix.

## Direction Encoding Contract

Canonical SPLT direction lists use initiator-relative `{0, 1}` values derived
from packet endpoints: `0` is source-to-destination and `1` is
destination-to-source. They contain no padding. The feature builder maps raw `0` to `-1`
and raw `1` to `+1` for both flattened and sequential SPLT representations;
padded positions use `0`. This prevents a valid direction value from colliding
with padding.

## Feature Representations

`configs/features.yaml` freezes the primary prefix at 50 packets, the ablation
prefixes at `{10, 20, 50, 80}`, three SPLT channels, timestep-major flattening,
and deterministic `log1p` transforms for size and timing. MatchedFlowStats and
PrefixStats retain raw measurement units and defer median imputation to each
training fold.

PrefixStats reconstructs packet timestamps from global inter-arrival times before
computing direction-specific gaps. Raw direction `0` is source-to-destination and
raw direction `1` is destination-to-source. Standard deviations use NFStream's
sample convention (`ddof=1`), with zero for fewer than two observations.

Audit every representation over the complete canonical dataset with:

```bash
vpncat-audit-features --config configs/features.yaml
```

The audit verifies that flattened SPLT is exactly the timestep-major reshape of
the sequential tensor and writes `data/processed/feature_audit.json`.

## Fold-Safe Preprocessing

`configs/preprocessing.yaml` freezes the preprocessing contract. For each of
the five folds, canonical rows are joined to the fixed split manifest by
`pair_id`; train, validation, and test roles must form a complete disjoint
partition. Both views of a physical flow inherit the same role because a pair is
represented by one canonical row.

For MatchedFlowStats and PrefixStats, missing values are imputed with medians
fitted only on the selected training view and training-role pairs. No scaling is
applied because these representations are used by tree models. Label vocabulary
and balanced class weights are likewise derived from training-role pairs only.
Each fitted state records the count and SHA-256 digest of its training `pair_id`
set. The opposite view, validation pairs, and test pairs cannot influence fitted
state.

Flattened and sequential SPLT use no data-fitted feature transformation: their
direction remapping, `log1p` magnitude transforms, padding, and masks are fixed
by the fixed representation contract. Neural models still consume the
training-only label vocabulary and class weights.

Run the complete preprocessing audit with:

```bash
vpncat-audit-preprocessing --config configs/preprocessing.yaml
```

The audit checks canonical and split hashes against the dataset manifest, then
fits both statistical representations from both training domains in all five
folds. It transforms train, validation, and test roles in both views, verifies
finite output and balanced class totals, and writes
`data/processed/preprocessing_audit.json`.

## Primary Run Contract

`configs/primary.yaml` enumerates the frozen nine representation/model
configurations. Across five folds, two training domains, one classical seed, and
three neural seeds, it expands to 150 unique trained-model runs. Every trained
model must predict both inner and outer views of the identical held-out pair set,
yielding 300 test-domain prediction groups without redundant retraining.

A run identity is defined by protocol, representation, model, fold, training
domain, and seed. Its output directory is platform-independent and derived from
that identity. Statistical runs hard-bind the applied fitted state's
`train_domain`, representation, fold, training-pair count, and training-pair hash
to the run. Flattened and sequential SPLT runs bind the audited fold target state
and explicitly record their fit-free feature transform and training domain.

Run publication is atomic and non-overwriting. A completed run contains:

```text
run.json
split_manifest.csv
metrics.json
metrics_long.csv
predictions.parquet
training_history.csv  # neural runs only
```

Predictions must cover every held-out `pair_id` exactly once per test domain and
must preserve session, label, fold, seed, model, representation, training domain,
and test domain. Probabilities must be finite, sum to one, and agree with the
stored prediction. Metrics are recomputed from the prediction artifact using the
frozen accuracy, balanced accuracy, macro F1, weighted F1, and macro one-vs-rest
average-precision definitions.

Audit the complete matrix and preprocessing bindings with:

```bash
vpncat-audit-experiment-contract --config configs/primary.yaml
```

Use `--artifact-dir`, `--output-root`, and `--output` to relocate data and outputs
without editing configuration files.

## Classical Runs

Random Forest and XGBoost use the fixed inherited hyperparameters recorded in
`configs/primary.yaml`; there is no fold-specific tuning. Random Forest receives
the audited training-fold class weights, while XGBoost receives the equivalent
per-sample weights. Validation rows are not passed to either estimator.

The runner refuses stale contract audits and existing output directories. One
fit always publishes predictions for both held-out views. For example:

```bash
vpncat-run-primary-classical \
  --config configs/primary.yaml \
  --experiment-id matched_flow_stats__random_forest \
  --fold 1 \
  --train-domain inner \
  --seed 42
```

If `primary.yaml` changes deliberately, regenerate the contract audit before
running models.

## Neural Contract

`configs/neural.yaml` freezes the PyTorch implementation, weighted
cross-entropy, AdamW policy, stopping and scheduling rules, development fold,
and the exact twelve-trial search space. The search varies only learning rate,
batch size, dropout, and width. It uses fold 1's inner training and validation
views; selected configurations are applied unchanged to all folds,
seeds, protocols, and training domains.

The three fixed topologies isolate model-family differences:

- CNN1D uses parallel kernels `{3, 7, 11}`, one residual convolutional block,
  and masked mean pooling. It contains no attention layer.
- LSTM uses two unidirectional recurrent layers and masked mean pooling. It
  contains no attention layer.
- Transformer uses two pre-normalized encoder layers, four attention heads,
  learnable packet positions, and masked mean pooling.

All models consume explicit boolean masks and return logits. Padded input is
zeroed before processing, masked positions are excluded from pooling and
attention, and the LSTM uses packed sequences. No topology applies softmax
before the loss. No augmentation, focal loss, mixed precision, or target-view
validation is permitted by the frozen neural configuration.

## Neural Tuning

Neural tuning requires a clean Git revision and a current primary experiment
contract audit. It reads only the inner/source sequence columns, materializes
only fold 1 training and validation tensors, and never constructs held-out test
tensors or outer-view features. Every trial resets the same seed before model
construction and data-loader creation.

Run all twelve trials for one architecture with:

```bash
vpncat-tune-neural --model cnn1d --device cuda
```

Run or resume selected trials independently with repeatable `--trial-id`
arguments:

```bash
vpncat-tune-neural --model cnn1d --device cuda --trial-id 1 --trial-id 2
```

Each completed trial is published atomically under
`outputs/tuning/<model>/trial_<id>/` with `trial.json` and
`training_history.csv`. Existing trials are reused only when their model,
parameters, input hashes, Git revision, and history hash all validate. Selection
is written only after all twelve trials are complete, using maximum validation
macro F1 with the lowest trial ID as the deterministic exact-tie rule.
Selection also requires every trial for an architecture to use the identical
device, platform, and recorded package environment.

The shared trainer uses training-fold balanced class weights exactly once,
standard mean-normalized weighted cross-entropy, AdamW, gradient clipping,
`ReduceLROnPlateau` on validation loss, and early stopping after six epochs
without a strict validation macro-F1 improvement. The best macro-F1 checkpoint
is restored in memory. No held-out prediction or metric is computed during
tuning.

## Primary Neural Runs

Primary neural execution requires all twelve valid tuning trials and complete
`selected.json` and `tuning_manifest.json` artifacts for the requested model.
Before every primary run, the runner revalidates every trial, recomputes the
selection from its validation history, and requires both selection artifacts to
match that result. Stale input hashes, a different Git revision, incomplete
trials, modified histories, and parameter-count drift are rejected.

The selected hyperparameters are reused unchanged for all folds, seeds, and
training domains. Each invocation trains one model on the source training view,
uses only the corresponding source validation view for scheduling and early
stopping, restores the best checkpoint, and predicts both held-out views of the
same pair-disjoint test flows:

```bash
vpncat-run-primary-neural \
  --model cnn1d \
  --fold 1 \
  --train-domain inner \
  --seed 42 \
  --device cuda
```

The completed run binds the hashes of `neural.yaml`, `selected.json`, and
`tuning_manifest.json` in `run.json`. It also records the selected trial, frozen
topology and policies, parameter count, best epoch, validation score, device,
runtime package versions, CUDA runtime, cuDNN version, and GPU model where
available. Predictions are generated in canonical pair order, apply softmax
exactly once to model logits, and are published through the same metric and
atomic-artifact contract used by classical runs.

## Primary Matrix Orchestration

`vpncat-primary-matrix` is the resumable controller for the frozen 150-run
primary matrix. Without `--execute`, it plans the selected subset and validates
every existing output. Missing run directories are reported as pending. An
existing directory is accepted only when its run identity, package version,
clean Git revision, input and tuning hashes, artifact inventory, predictions,
and recomputed metrics all validate; incompatible or partial output stops the
entire preflight before any model is trained.

Filters are repeatable and may select family, model, representation, fold,
training domain, seed, or exact run ID. For example, inspect all CNN1D runs:

```bash
vpncat-primary-matrix --model cnn1d
```

After tuning has completed and been audited, execute or resume that subset on
Apple Silicon:

```bash
vpncat-primary-matrix --model cnn1d --execute --device mps
```

Limit execution to one pending run for a post-tuning smoke test:

```bash
vpncat-primary-matrix \
  --model cnn1d \
  --fold 1 \
  --train-domain inner \
  --seed 42 \
  --execute \
  --maximum-pending-runs 1 \
  --device mps \
  --report outputs/primary-smoke-report.json
```

The controller preloads and validates all required neural selections before
training. Trials within each model and selected models within one batch must
share an execution environment and device. A selected configuration is loaded
once per model and passed unchanged to every run. Each newly published run is
immediately revalidated through the same current-revision contract before the
batch proceeds.

On every platform, XGBoost runs execute in a fresh Python subprocess. Classical
CLI and artifact paths never import the PyTorch runtime; they record its package
version through metadata only. This isolation avoids duplicate OpenMP runtime
collisions on macOS while allowing a single matrix invocation to include both
classical and neural runs.

## Cross-Session Contract

`configs/cross_session.yaml` freezes two directional evaluations: train on
Session 1 and test on Session 2, then train on Session 2 and test on Session 1.
Only inner views of source-session pairs are used for training and validation;
each fitted model later predicts both inner and outer views of every target-
session pair. The base matrix inherits all nine primary configurations and
contains 30 trainings: 12 classical and 18 neural.

For each direction, exactly 10% of source-session pairs are assigned to
validation. Class quotas use deterministic largest-remainder allocation, and
pairs within a class are selected by a seeded SHA-256 rank. This makes the split
independent of canonical input row order and library shuffle implementations.
All target-session pairs are test-only, and all 14 classes must occur in every
train, validation, and test role.

Build the split and machine-readable run contract with:

```bash
vpncat-cross-session-contract --config configs/cross_session.yaml
```

Validate existing artifacts by reconstructing the split from canonical data:

```bash
vpncat-cross-session-contract \
  --config configs/cross_session.yaml \
  --validate
```

The supervised outer-trained cross-session references are not part of this base
matrix. Their configurations remain explicitly deferred until the primary
experiment selects the strongest models using the predeclared criterion.

## Cross-Session Preprocessing

`configs/cross_session_preprocessing.yaml` freezes source-only preprocessing for
both cross-session directions. Statistical medians, label vocabulary, and
balanced class weights are fitted only from the inner view of training-role
pairs in the source session. Source validation pairs, every target-session pair,
and the complete outer view are excluded from fitting. The fitted state records
the exact count and SHA-256 digest of its training-pair set.

MatchedFlowStats and PrefixStats use training-median imputation without scaling.
Flattened and sequential SPLT retain their deterministic, fit-free transforms;
only their target vocabulary and class weights are fitted. Materializers expose
source-inner training and validation arrays and paired inner/outer test arrays
for every target-session pair.

Run the leakage and transformation audit after building the cross-session
contract:

```bash
vpncat-audit-cross-session-preprocessing \
  --config configs/cross_session_preprocessing.yaml
```

The audit reconstructs both directional indexes, checks balanced source-only
target state, transforms every role in both views, and adversarially poisons
validation, target-session, and opposite-domain values. It writes
`data/processed/cross_session_preprocessing_audit.json` bound to the canonical,
split, contract-audit, configuration, package, and Git revisions.

## Cross-Session Classical Runs

Random Forest and XGBoost inherit the fixed primary hyperparameters. Each run
trains on only the inner view of source-session training pairs; source validation
pairs are excluded from estimator fitting. The fitted model predicts both views
of every target-session pair and publishes them under one run identity.

```bash
vpncat-run-cross-session-classical \
  --config configs/cross_session_preprocessing.yaml \
  --experiment-id matched_flow_stats__random_forest \
  --train-session 1 \
  --seed 42
```

Cross-session predictions explicitly record `train_session`, `test_session`,
`train_domain`, and `test_domain`; they do not invent a fold identifier. Before
atomic publication, the writer verifies exact paired target coverage,
session/label agreement, finite probability-simplex values, prediction argmax,
all five frozen metrics, and every artifact hash. Existing run directories are
never overwritten. `--artifact-dir` and `--output-root` relocate inputs and
outputs without source edits.

## Cross-Session Neural Runs

CNN1D, LSTM, and Transformer reuse their selected primary hyperparameters
unchanged. Each run trains on source-session inner training pairs, uses only the
source-session inner validation subset for early stopping and scheduling, then
predicts both views of every target-session pair with the same trained model.

```bash
vpncat-run-cross-session-neural \
  --config configs/cross_session_preprocessing.yaml \
  --neural-config configs/neural.yaml \
  --model cnn1d \
  --train-session 1 \
  --seed 42 \
  --device cuda
```

The selected tuning result, tuning manifest, and neural configuration are
content-hashed into every run. The manifest also records the selected trial,
fixed topology, optimizer and training policies, parameter count, best epoch,
validation score, execution device, and tuning environment. Neural publication
requires a finite `training_history.csv` with the frozen schema; completed runs
cannot be validated without supplying the exact expected selection hashes.

## Cross-Session Matrix

The cross-session controller preflights, validates, filters, and resumes all 30
base runs. Planning does not require tuning selections when no neural output is
being executed or revalidated. Execution and existing neural outputs require
complete selections for the current clean Git revision.

```bash
vpncat-cross-session-matrix \
  --config configs/cross_session_preprocessing.yaml \
  --neural-config configs/neural.yaml \
  --report outputs/cross-session-plan.json
```

Add `--execute` to run pending entries. Filters include family, model,
representation, training session, seed, and exact run ID; use
`--maximum-pending-runs 1` for a bounded smoke test. Every new output is
immediately revalidated against its source-pair state, selection hashes,
predictions, metrics, artifact inventory, and current clean revision before the
controller advances. Incompatible existing output stops the campaign before
any pending run executes. XGBoost always runs in an isolated subprocess, while
the classical orchestration import path remains PyTorch-free.

## DANN Contract

`configs/dann.yaml` freezes 15 CNN1D-backed domain-adaptation runs across five
folds and three seeds. Each run pairs labeled inner training tensors with
unlabeled outer tensors from the exact same ordered training `pair_id` set.
Outer validation and test pairs are prohibited from adaptation; early stopping
and learning-rate scheduling use only the source-inner validation subset.

```bash
vpncat-dann-contract --config configs/dann.yaml
```

The contract fixes the standard logistic gradient-reversal ramp from 0 toward 1 with
`gamma=10`, paired source/target batches, unit domain-loss weight, one hidden
domain-head layer, the selected primary CNN1D width/dropout, and no
augmentation. The adaptation subset has no target-label field by construction.
Both held-out views are retained for paired diagnostics, while outer-view
classification is the primary adaptation endpoint. The audit pins the primary
data, split, preprocessing and run contracts, neural configuration, every role
pair-set digest, and all 15 run identities.

The DANN model reuses the frozen CNN1D encoder and classification head selected
by the primary neural search. A binary domain head receives the pooled encoder
representation through gradient reversal. One deterministic loader applies the
same shuffle to each matched inner/outer training pair; weighted classification
loss uses only inner labels, while binary domain loss uses balanced inner and
outer examples. Checkpoint selection and learning-rate scheduling use only the
inner validation subset. Schedule progress is measured against the frozen
maximum optimization-step budget; the realized coefficient at every epoch
boundary is stored in training history, including when early stopping ends a
run before the maximum budget.

Run or preflight the complete DANN matrix after CNN1D tuning is complete:

```bash
vpncat-dann-matrix --config configs/dann.yaml
vpncat-dann-matrix --config configs/dann.yaml --execute --device auto
```

Use `--fold`, `--seed`, and `--maximum-pending-runs` to filter or bound a
campaign. Plan-only preflight validates the clean revision, input chain, DANN
contract, and all selected run identities without loading tuning artifacts or
PyTorch when every run is pending. Each atomic run directory records the exact
tuning selection, model and domain-adaptation policies, pair-set hashes,
training history, paired predictions, and all five frozen metrics. Existing
outputs are revalidated before a campaign continues.

A single DANN run can also be executed directly:

```bash
vpncat-run-dann --config configs/dann.yaml --fold 1 --seed 42 --device auto
```

## Representation Ablation Contracts

The two ablation contracts vary observations while keeping the selected CNN1D
and Transformer architectures fixed. `configs/ablation_prefix.yaml` defines
`N={10,20,50,80}` with all channels. `configs/ablation_channels.yaml` defines
five channel combinations at `N=50`, including size plus timing without
direction. Every executable run trains on inner views only and predicts both
held-out views using seed 42.

```bash
vpncat-ablation-contract --config configs/ablation_prefix.yaml
vpncat-ablation-contract --config configs/ablation_channels.yaml
```

The `N=50`, all-channel cells are references to the matching primary CNN1D and
Transformer seed-42 artifacts, not new trainings. Across both tables there are
90 cells: 70 executable ablation runs and 20 references to the same 10 primary
artifacts. Reference cells are rejected by the ablation data materializer so
the baseline cannot be silently retrained.

After the contracts and neural tuning selections have been materialized, plan
or resume either matrix with:

```bash
vpncat-ablation-matrix --config configs/ablation_prefix.yaml
vpncat-ablation-matrix --config configs/ablation_channels.yaml
vpncat-ablation-matrix --config configs/ablation_prefix.yaml --execute --device cuda
```

Preflight reports executable and primary-reference cells separately. Existing
outputs are content-validated against their contract, pair sets, tuning
selection, metrics, and clean Git revision. Primary references are never added
to the execution set.

A single ablation cell can also be executed directly (the observation id names
one prefix-length or channel condition from the contract):

```bash
vpncat-run-ablation --config configs/ablation_prefix.yaml \
  --model cnn1d --observation n020 --fold 1 --device auto
```

Prefix observation ids are `n010`, `n020`, `n050`, and `n080`; channel
observation ids are `direction`, `direction_size`, `direction_timing`,
`size_timing`, and `all`.

## Analysis Inventory

`configs/analysis.yaml` freezes the five metrics, seed-ensemble policy, and
paired 1,000-replicate bootstrap policy before result aggregation. Its contract
enumerates every unique physical run artifact across the primary,
cross-session, DANN, and ablation protocols:

```bash
vpncat-analysis-contract
```

The 20 logical ablation anchors reference 10 primary seed-42 artifacts and are
not counted as additional physical outputs. The inventory pins every upstream
protocol contract and exact expected prediction-row count, preventing reused
anchors from being double-counted during analysis.

The seed-ensemble planner collapses the inventory into 44 physical prediction
groups and 46 logical paper groups. Ten groups average three neural seeds at the
class-probability level before taking `argmax`; all classical and ablation
groups use their single frozen seed. The two CNN1D/Transformer ablation anchors
are materialized once each and exposed through separate prefix and channel
aliases.

After every protocol controller reports complete, publish the final
machine-readable result bundle with:

```bash
vpncat-aggregate-results
```

Aggregation refuses a dirty tree, stale run, incomplete campaign, or existing
analysis directory. It writes deduplicated probability partitions, logical
alias mappings, five point metrics, separate per-seed metrics and dispersion,
and paired bootstrap intervals for balanced accuracy, macro F1, and their
outer-minus-inner encapsulation gaps.

## Tests

```bash
pytest
ruff check .
```

Tests use synthetic paired data and do not require the downloaded dataset.

## License

MIT. See `LICENSE`.
