# 🔊 Anomalous Sound Detection — Pseudo-MLOps Pipeline

> An unsupervised anomaly detection system for industrial machine sounds, built with a lightweight pseudo-MLOps architecture. Detects abnormal gearbox sounds using a Conv2D Autoencoder trained on mel spectrograms.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [MLOps Features](#mlops-features)
- [Setup Guide](#setup-guide)
- [Running on Google Colab](#running-on-google-colab)
- [Pipeline Execution](#pipeline-execution)
- [Configuration](#configuration)
- [Artifact Versioning](#artifact-versioning)
- [Experiment Tracking](#experiment-tracking)
- [File Responsibilities](#file-responsibilities)
- [Build Roadmap](#build-roadmap)

---

## 🧠 Project Overview

This project detects anomalous sounds in industrial machinery (gearbox) using an **unsupervised autoencoder-based approach**. No labeled anomaly data is needed during training — the model learns what "normal" sounds like and flags anything that deviates significantly.

| Property | Detail |
|---|---|
| **Task** | Unsupervised Anomaly Detection |
| **Domain** | Industrial Sound / Predictive Maintenance |
| **Model** | Conv2D Autoencoder |
| **Input** | Raw `.wav` audio files |
| **Output** | Anomaly score + severity label (Normal / Medium / Severe) |
| **Framework** | TensorFlow / Keras |
| **MLOps Tools** | MLflow, YAML configs, versioned artifacts |
| **Compute** | Google Colab (T4 GPU) + Google Drive storage |

---

## ⚙️ How It Works

```
Raw .wav Audio
      ↓
audio_loader.py     →    Load + Resample to 16kHz
      ↓
spectrogram.py      →    STFT → Mel Scale → dB → Save as .png
      ↓
dataset.py          →    Wrap PNGs into tf.data.Dataset
      ↓
autoencoder.py      →    Conv2D Encoder-Decoder (trained on normal sounds only)
      ↓
anomaly_scorer.py   →    Reconstruction Error → Mahalanobis + PCA Score
      ↓
classifier.py       →    Score → Normal / Medium / Severe
```

### Core Idea

- Autoencoder is trained **exclusively on normal sounds**
- It learns to reconstruct normal audio well
- When **abnormal audio** is passed through, reconstruction error is **high**
- That error becomes the anomaly score

---

## 📁 Project Structure

```
anomalous-sound-detection/
│
├── data/
│   ├── raw/
│   │   └── gearbox/
│   │       ├── train/          # normal sounds only (.wav)
│   │       ├── source_test/    # test audio
│   │       └── target_test/
│   └── processed/
│       ├── train/              # mel spectrograms (.png)
│       ├── source_test/
│       └── target_test/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audio_loader.py     # .wav → numpy audio array
│   │   ├── spectrogram.py      # audio → mel spectrogram → .png
│   │   └── dataset.py          # .png → tf.data / torch Dataset
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       # abstract model interface
│   │   └── autoencoder.py      # Conv2D encoder-decoder
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # fit loop + MLflow + seed + versioning
│   │   └── losses.py           # MSE + custom loss functions
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── anomaly_scorer.py   # Mahalanobis + PCA scoring
│   │   ├── classifier.py       # score → severity label
│   │   └── predictor.py        # end-to-end single file prediction
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # console + timestamped file logging
│       ├── versioning.py       # auto v1/v2/v3 artifact versioning
│       ├── metrics.py          # AUC, pAUC calculation
│       └── visualization.py   # spectrogram + score plots (notebooks only)
│
├── pipelines/
│   ├── 01_preprocess.py        # raw audio → spectrograms
│   ├── 02_train.py             # train autoencoder
│   ├── 03_evaluate.py          # compute AUC / pAUC on test set
│   └── 04_predict.py           # predict on new audio file
│
├── configs/
│   ├── default.yaml            # all paths, hyperparams, mlflow, seed
│   └── experiment_01.yaml      # override file for specific experiments
│
├── artifacts/
│   ├── models/
│   │   ├── v1/                 # versioned model checkpoints
│   │   ├── v2/
│   │   └── latest -> v2/       # symlink to latest version
│   ├── metadata/
│   │   ├── run_v1.json         # params + metrics per run
│   │   └── run_v2.json
│   └── logs/
│       └── run_20240601_143022.log
│
├── notebooks/
│   ├── 00_setup.ipynb          # drive mount + installs + sys.path
│   ├── 01_preprocess.ipynb     # EDA + audio exploration
│   ├── 02_train.ipynb          # training + monitoring
│   ├── 03_evaluate.ipynb       # evaluation + score analysis
│   ├── 04_predict.ipynb        # single file prediction
│   └── 05_full_pipeline.ipynb  # end-to-end run
│
├── tests/
│   ├── test_spectrogram.py
│   ├── test_model.py
│   └── test_scorer.py
│
├── mlruns/                     # auto-created by MLflow
├── run_pipeline.py             # master orchestrator script
├── Makefile                    # command shortcuts
├── setup.py                    # makes src/ importable as package
├── requirements.txt            # pinned dependencies
├── environment.yml             # conda environment definition
└── README.md
```

---

## 🏗️ MLOps Features

| MLOps Principle | Implementation |
|---|---|
| **Experiment Tracking** | MLflow logs params, metrics, and artifacts per run |
| **Artifact Versioning** | Auto-increments `v1/`, `v2/`, `v3/` folders |
| **Config-Driven** | All settings in `configs/*.yaml` — zero hardcoding |
| **Reproducibility** | Global seed set in config, applied to all libraries |
| **Structured Logging** | Timestamped log files saved to `artifacts/logs/` |
| **Metadata Tracking** | JSON saved per run in `artifacts/metadata/` |
| **Pipeline Orchestration** | `run_pipeline.py` chains all 4 stages in order |
| **Modular Architecture** | Each module has a single clear responsibility |
| **Testability** | Independent unit tests per module |

---

## 🛠️ Setup Guide

### Option A — Local Machine

```bash
# 1. Clone the repository
git clone https://github.com/yourname/anomalous-sound-detection.git
cd anomalous-sound-detection

# 2. Create conda environment
conda env create -f environment.yml
conda activate anomaly-detection

# 3. Install src as importable package
pip install -e .

# 4. Verify setup
python -c "from src.utils.logger import get_logger; print('Setup OK')"
```

### Option B — pip only

```bash
pip install -r requirements.txt
pip install -e .
```

---

## ☁️ Running on Google Colab

This project is designed to run on **Google Colab with Google Drive** as persistent storage.

> **Strategy: Google Drive = Storage | Colab = Compute**

### Every Session — Run These 3 Cells First

**Cell 1 — Mount Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 — Set Project Root**
```python
import os, sys
PROJECT_ROOT = "/content/drive/MyDrive/anomalous-sound-detection"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
```

**Cell 3 — Install Dependencies**
```python
!pip install -q librosa mlflow pyyaml scikit-learn tensorflow
```

### Enable GPU

`Runtime → Change Runtime Type → T4 GPU`

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Should show GPU
```

### MLflow UI on Colab

```python
!pip install -q pyngrok
from pyngrok import ngrok
!mlflow ui --port 5000 &
print(ngrok.connect(5000))   # Opens public URL
```

---

## 🚀 Pipeline Execution

### Run Full Pipeline (Recommended)

```bash
python run_pipeline.py
# or
make all
```

### Run Individual Stages

```bash
# Stage 1 — Preprocess raw audio → spectrograms
python pipelines/01_preprocess.py

# Stage 2 — Train autoencoder
python pipelines/02_train.py

# Stage 3 — Evaluate on test set
python pipelines/03_evaluate.py

# Stage 4 — Predict on new audio
python pipelines/04_predict.py --file data/raw/gearbox/target_test/sample.wav
```

### Makefile Shortcuts

```bash
make preprocess    # run stage 1
make train         # run stage 2
make evaluate      # run stage 3
make predict       # run stage 4
make all           # run full pipeline
make test          # run all tests
make clean         # clear pycache
```

---

## ⚙️ Configuration

All settings live in `configs/default.yaml`. **Never hardcode values in Python files.**

```yaml
paths:
  data_raw:        data/raw/gearbox
  data_processed:  data/processed
  artifacts:       artifacts

versioning:
  mode: auto            # auto = increment version, manual = use below
  manual_version: v1

mlflow:
  experiment_name: anomaly-detection
  tracking_uri: mlruns

training:
  epochs:        50
  batch_size:    32
  learning_rate: 0.001
  seed:          42

inference:
  threshold_normal: 0.5
  threshold_severe: 0.8
```

### Running a Custom Experiment

Create `configs/experiment_01.yaml` with only the values you want to override:

```yaml
training:
  learning_rate: 0.0005
  epochs: 100

mlflow:
  experiment_name: anomaly-detection-exp01
```

Then run:

```bash
python pipelines/02_train.py --config configs/experiment_01.yaml
```

---

## 📦 Artifact Versioning

Every training run creates a new versioned folder automatically:

```
artifacts/
└── models/
    ├── v1/                   # first run
    │   └── autoencoder.h5
    ├── v2/                   # second run
    │   └── autoencoder.h5
    └── latest -> v2/         # always points to newest
```

Version is auto-incremented by scanning existing folders. To use a specific version set `versioning.mode: manual` in config.

---

## 📊 Experiment Tracking

Each run generates:

**MLflow** — logged to `mlruns/` (view with `mlflow ui`)
- Parameters: learning rate, epochs, batch size, seed, version
- Metrics: train loss, val loss per epoch
- Artifacts: model file, metadata JSON

**Metadata JSON** — saved to `artifacts/metadata/run_vN.json`

```json
{
    "version": "v1",
    "run_id": "abc123def456",
    "timestamp": "2024-06-01T14:30:22",
    "params": {
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "seed": 42
    },
    "metrics": {
        "final_val_loss": 0.003421
    },
    "artifacts": {
        "model_path": "artifacts/models/v1/autoencoder.h5"
    }
}
```

**Log File** — saved to `artifacts/logs/run_YYYYMMDD_HHMMSS.log`

---

## 📋 File Responsibilities

| File | Single Job |
|---|---|
| `audio_loader.py` | `.wav` path → clean numpy audio array |
| `spectrogram.py` | Audio array → mel spectrogram → saved `.png` |
| `dataset.py` | `.png` files → batched tf.data / torch Dataset |
| `autoencoder.py` | Define Conv2D encoder-decoder architecture |
| `trainer.py` | Fit loop + MLflow logging + seed + save artifacts |
| `losses.py` | MSE + custom loss function definitions |
| `anomaly_scorer.py` | Reconstruction error → Mahalanobis + PCA score |
| `classifier.py` | Score → Normal / Medium / Severe label |
| `predictor.py` | End-to-end prediction on a single audio file |
| `logger.py` | Console + timestamped file logging |
| `versioning.py` | Auto v1/v2/v3 version resolution |
| `metrics.py` | AUC and pAUC calculation |
| `visualization.py` | Plot spectrograms and scores in notebooks |

---

## 🗺️ Build Roadmap

```
Phase 1  →  Environment setup + imports verified
Phase 2  →  configs/default.yaml written
Phase 3  →  Utils — logger, versioning, metrics, visualization
Phase 4  →  Data layer — audio_loader, spectrogram, dataset, 01_preprocess
Phase 5  →  Model — base_model, autoencoder
Phase 6  →  Training — losses, trainer, 02_train + MLflow verified
Phase 7  →  Inference — scorer, classifier, predictor, 03_evaluate, 04_predict
Phase 8  →  Orchestration — run_pipeline.py + Makefile
Phase 9  →  Tests — test_spectrogram, test_model, test_scorer
Phase 10 →  Full clean run verification
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_spectrogram.py -v
```

---

## 📦 Requirements

```
tensorflow>=2.12.0
librosa>=0.10.1
numpy>=1.23.0
scikit-learn>=1.3.0
mlflow>=2.5.0
pyyaml>=6.0
matplotlib>=3.7.0
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Follow existing module structure and naming conventions
4. Add tests for any new functionality
5. Submit a pull request

---

## 📄 License

MIT License — see `LICENSE` file for details.

---

## 👤 Author

Built as a portfolio project demonstrating pseudo-MLOps practices on a real-world audio anomaly detection problem.