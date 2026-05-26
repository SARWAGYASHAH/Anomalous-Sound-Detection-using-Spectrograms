# Anomalous Sound Detection

Keras autoencoder pipeline for detecting anomalous industrial machine sounds
from mel spectrograms. The current project is focused on the ML workflow only:
preprocessing, training, evaluation, and single-file prediction.

## Current Status

- TensorFlow/Keras model foundation and Conv2D autoencoder are implemented.
- Training runs through `model.compile()` and `model.fit()` with checkpointing,
  early stopping, MLflow logging, versioned model folders, and metadata.
- Full training is intended for Google Colab; local execution is for tests and
  dry checks.
- Reconstruction-error scoring, threshold helpers, severity classification,
  evaluation metrics, and WAV prediction are implemented.
- Optional Mahalanobis helper functions remain available in
  `src/inference/anomaly_scorer.py` for future scoring experiments.
- No backend or user interface is part of this phase.

## Flow

```text
.wav audio
  -> AudioLoader
  -> mel SpectrogramExtractor: (128, 313, 1)
  -> Conv2D Keras autoencoder
  -> reconstruction MSE score
  -> train-normal threshold
  -> normal / follow_up / alert
```

## Project Layout

```text
config/
  default.yaml              Base settings
  experiment_01.yaml        Experiment overrides
pipeline/
  01_preprocess.py          WAV files to processed .npy spectrograms
  02_train.py               Keras training and model artifacts
  03_evaluate.py            Metrics, plots, and evaluation metadata
  04_predict.py             Single WAV prediction
src/
  data/                     Audio, spectrogram, and tf.data utilities
  models/                   Keras base model and Conv2D autoencoder
  training/                 Losses and Keras trainer
  inference/                Scoring, severity classification, prediction
  utils/                    Logging, metrics, seeds, metadata, plots
tests/                      Lightweight shape and inference tests
notebooks/
  06_keras_colab_training.ipynb
artifacts/                  Generated models, metadata, plots, and logs
run_pipeline.py             Command-line stage orchestrator
```

Processed data is expected below:

```text
Data/processed/
  train/normal/*.npy
  source_test/normal/*.npy
  source_test/anomaly/*.npy
  target_test/normal/*.npy
  target_test/anomaly/*.npy
```

The preprocessing stage reads raw WAV files from `Data/gearbox/` by default,
based on `data.machine_type` in `config/default.yaml`.

## Setup

Local environment:

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

Colab can use the lightweight requirements file because TensorFlow is already
available in the runtime:

```bash
pip install -r requirements-colab.txt
pip install -e .
```

## Local Checks

Preprocess the raw dataset:

```bash
python pipeline/01_preprocess.py --config config/default.yaml
```

Build datasets and the Keras model and run one batch without training:

```bash
python pipeline/02_train.py --config config/default.yaml --dry-run --no-mlflow
```

Evaluate an existing saved model:

```bash
python pipeline/03_evaluate.py --config config/default.yaml --model-path artifacts/models/v2/best_model.keras
```

Predict one WAV file:

```bash
python pipeline/04_predict.py --config config/default.yaml --model-path artifacts/models/v2/best_model.keras --file Data/gearbox/source_test/section_00_source_test_normal_0000.wav
```

When `--model-path` is omitted, evaluation and prediction select the newest
available `.keras` artifact automatically.

## Colab Training

Use `notebooks/06_keras_colab_training.ipynb` to mount Drive, install
dependencies, preprocess or restore processed files, and run training:

```bash
python pipeline/02_train.py --config config/default.yaml
python pipeline/03_evaluate.py --config config/default.yaml
```

Each real training run writes a versioned directory:

```text
artifacts/models/v1/
  model.keras               Canonical saved model
  best_model.keras          Lowest monitored loss checkpoint
  final_model.keras         Final trainer save
  config_snapshot.yaml
  training_log.csv
```

Training metadata is written to `artifacts/metadata/`; MLflow files are written
under `mlruns/` when `mlflow.enabled` is true.

## Orchestration

The orchestrator prevents accidental full local training unless explicitly
enabled.

Local one-batch workflow:

```bash
python run_pipeline.py --config config/default.yaml --stages train --train-dry-run
python run_pipeline.py --config config/default.yaml --stages evaluate predict --model-path artifacts/models/v2/best_model.keras --predict-file path/to/file.wav
```

Full Colab workflow:

```bash
python run_pipeline.py --config config/default.yaml --allow-training
```

Add `--predict-file path/to/file.wav` to run prediction after evaluation.

Make shortcuts:

```bash
make train-dry
make evaluate MODEL=artifacts/models/v2/best_model.keras
make predict MODEL=artifacts/models/v2/best_model.keras AUDIO=path/to/file.wav
make pipeline-colab
```

## Evaluation Artifacts

`pipeline/03_evaluate.py` fits the anomaly threshold using processed normal
training samples and evaluates labeled test samples. It saves:

```text
artifacts/evaluation/source_test/
  metrics.json
  scores.csv
  anomaly_score_distribution.png
  roc_curve.png
artifacts/metadata/run_evaluation_*.json
```

Reported metrics include AUC, partial AUC, average precision, precision,
recall, F1, and confusion matrix counts.

## Configuration

Important settings in `config/default.yaml`:

```yaml
model:
  autoencoder:
    input_dim: 128
    latent_dim: 32

inference:
  threshold_method: percentile
  percentile: 95
  classification:
    labels: [normal, follow_up, alert]
    boundaries: [0.5, 0.8]
```

Severity boundaries apply to score divided by the learned anomaly threshold:
scores reaching the threshold set `crosses_threshold` for strict metric
reporting, while lower relative scores can still be marked as requiring
attention by the severity policy.
