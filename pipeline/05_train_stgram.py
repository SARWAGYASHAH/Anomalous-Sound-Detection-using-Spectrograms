"""Train STgram-MFN and fit per-section GMM anomaly scorers.

Usage:
    python pipeline/05_train_stgram.py --config config/stgram_mfn.yaml
    python pipeline/05_train_stgram.py --epochs 20 --batch-size 16
    python pipeline/05_train_stgram.py --pretrained-checkpoint path/to/model.pt --freeze-backbone
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.stgram_dataset import (  # noqa: E402
    STgramWaveDataset,
    build_section_label_map,
    discover_wav_files,
    resolve_wav_split_dir,
)
from src.models.stgram_mfn import build_stgram_mfn, freeze_feature_extractor  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.visualization import plot_loss_curve  # noqa: E402


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(base_path: str, override_path: str | None) -> dict[str, Any]:
    with open(base_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if override_path and Path(override_path).resolve() != Path(base_path).resolve():
        with open(override_path, "r", encoding="utf-8") as handle:
            config = deep_merge(config, yaml.safe_load(handle))
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if args.epochs is not None:
        updated["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        updated["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        updated["training"]["learning_rate"] = args.learning_rate
    if args.output_dir is not None:
        updated.setdefault("stgram", {})["output_dir"] = args.output_dir
    if args.pretrained_checkpoint is not None:
        updated.setdefault("stgram", {})["pretrained_checkpoint"] = args.pretrained_checkpoint
    if args.freeze_backbone:
        updated.setdefault("stgram", {})["freeze_backbone"] = True
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STgram-MFN on raw WAV files")
    parser.add_argument("--config", default="config/stgram_mfn.yaml", help="STgram config YAML.")
    parser.add_argument("--base-config", default="config/default.yaml", help="Base config YAML.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--output-dir", default=None, help="Override artifact root.")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Optional .pt checkpoint to fine-tune.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only ArcFace/classifier heads.")
    parser.add_argument("--dry-run", action="store_true", help="Build one batch and exit without training.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    return parser.parse_args()


def raw_split_dir(config: dict[str, Any], split: str) -> Path:
    stgram = config.get("stgram", {})
    explicit_key = f"raw_{split}_dir"
    return resolve_wav_split_dir(
        split=split,
        machine_type=config["data"].get("machine_type", "gearbox"),
        data_root=Path(config["data"].get("raw_dir", "Data/raw")).parent,
        explicit_dir=stgram.get(explicit_key),
    )


def make_run_dir(config: dict[str, Any]) -> Path:
    output_root = Path(config.get("stgram", {}).get("output_dir", "artifacts/models/stgram_mfn"))
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_root / "latest.txt"
    latest_path.write_text(str(run_dir), encoding="utf-8")
    return run_dir


def split_indices(count: int, train_split: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, int(count * train_split))
    train_count = min(train_count, count - 1) if count > 1 else count
    return indices[:train_count], indices[train_count:]


def create_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = str(config["training"].get("optimizer", "adam")).lower()
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler_config = config["training"].get("scheduler", {})
    scheduler_type = str(scheduler_config.get("type", "cosine")).lower()
    if scheduler_type == "none":
        return None
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 10)),
            gamma=float(scheduler_config.get("gamma", 0.5)),
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
        eta_min=float(scheduler_config.get("eta_min", 0.0)),
    )


def load_pretrained_if_requested(model, config: dict[str, Any], logger) -> None:
    checkpoint_path = config.get("stgram", {}).get("pretrained_checkpoint")
    if not checkpoint_path:
        return
    payload = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded pretrained checkpoint: %s", checkpoint_path)
    logger.info("Missing checkpoint keys: %s", missing)
    logger.info("Unexpected checkpoint keys: %s", unexpected)


def train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        waveforms = batch["waveform"].to(device, non_blocking=True)
        mels = batch["mel"].to(device, non_blocking=True)
        labels = batch["section_label"].to(device, non_blocking=True)

        logits, _ = model(waveforms, mels, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device) -> tuple[float, float]:
    if len(loader.dataset) == 0:
        return 0.0, 0.0
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="val", leave=False):
        waveforms = batch["waveform"].to(device, non_blocking=True)
        mels = batch["mel"].to(device, non_blocking=True)
        labels = batch["section_label"].to(device, non_blocking=True)

        logits, _ = model(waveforms, mels, labels)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def extract_features_by_section(model, loader, device) -> dict[int, np.ndarray]:
    model.eval()
    features_by_section: dict[int, list[np.ndarray]] = {}
    for batch in tqdm(loader, desc="extract_features"):
        waveforms = batch["waveform"].to(device, non_blocking=True)
        mels = batch["mel"].to(device, non_blocking=True)
        labels = batch["section_label"].numpy()
        features = model.extract_features(waveforms, mels).cpu().numpy()
        for feature, label in zip(features, labels):
            features_by_section.setdefault(int(label), []).append(feature)
    return {label: np.asarray(features, dtype=np.float32) for label, features in features_by_section.items()}


def save_checkpoint(path: Path, model, optimizer, config, section_to_label, history, epoch: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "section_to_label": section_to_label,
            "history": history,
            "epoch": epoch,
        },
        path,
    )


def load_torch_checkpoint(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def set_torch_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.base_config, args.config), args)
    set_torch_seed(int(config["seed"]))

    logger = get_logger("train_stgram", log_dir=config["artifacts"]["logs_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("PIPELINE STAGE 5: Training (STgram-MFN)")
    logger.info("=" * 60)
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("Device: %s", device)

    train_files = discover_wav_files(raw_split_dir(config, "train"))
    if not train_files:
        raise FileNotFoundError(f"No training WAV files found in {raw_split_dir(config, 'train')}")

    section_to_label = build_section_label_map(train_files)
    stgram_config = config["model"]["stgram_mfn"]
    dataset = STgramWaveDataset(
        train_files,
        section_to_label=section_to_label,
        sample_rate=int(config["audio"]["sample_rate"]),
        duration=float(config["audio"]["duration"]),
        n_fft=int(config["spectrogram"]["n_fft"]),
        hop_length=int(config["spectrogram"]["hop_length"]),
        n_mels=int(stgram_config.get("n_mels", config["spectrogram"]["n_mels"])),
        power=float(config["spectrogram"]["power"]),
        normalize_mel=bool(config.get("stgram", {}).get("normalize_mel", True)),
    )

    train_indices, val_indices = split_indices(len(dataset), float(config["data"]["train_split"]), int(config["seed"]))
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    workers = args.num_workers if args.num_workers is not None else int(config.get("stgram", {}).get("num_workers", 2))
    batch_size = int(config["training"]["batch_size"])
    train_loader = create_loader(train_subset, batch_size=batch_size, shuffle=True, workers=workers)
    val_loader = create_loader(val_subset, batch_size=batch_size, shuffle=False, workers=workers)

    model = build_stgram_mfn(stgram_config, num_classes=len(section_to_label)).to(device)
    load_pretrained_if_requested(model, config, logger)
    if bool(config.get("stgram", {}).get("freeze_backbone", False)):
        freeze_feature_extractor(model)
        logger.info("Frozen TgramNet and MobileFaceNet backbone. Training heads only.")

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    criterion = nn.CrossEntropyLoss()

    batch = next(iter(train_loader))
    with torch.no_grad():
        logits, features = model(
            batch["waveform"].to(device),
            batch["mel"].to(device),
            batch["section_label"].to(device),
        )
    logger.info("Training files: %s", len(train_files))
    logger.info("Section labels: %s", section_to_label)
    logger.info("Dry batch waveform: %s", tuple(batch["waveform"].shape))
    logger.info("Dry batch mel: %s", tuple(batch["mel"].shape))
    logger.info("Dry batch logits: %s", tuple(logits.shape))
    logger.info("Dry batch features: %s", tuple(features.shape))
    logger.info("Model parameters: %s", sum(parameter.numel() for parameter in model.parameters()))

    run_dir = make_run_dir(config)
    save_json(config, run_dir / "config_snapshot.json")
    save_json(section_to_label, run_dir / "section_to_label.json")

    if args.dry_run:
        logger.info("Dry run complete. Artifacts initialized at %s", run_dir)
        return

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    epochs = int(config["training"]["epochs"])
    best_path = run_dir / "best_model.pt"
    final_path = run_dir / "final_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": lr,
        }
        history.append(row)
        logger.info(
            "Epoch %03d/%03d train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f lr=%.6f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, config, section_to_label, history, epoch)

    save_checkpoint(final_path, model, optimizer, config, section_to_label, history, epochs)
    pd.DataFrame(history).to_csv(run_dir / "training_log.csv", index=False)
    plot_loss_curve(
        [row["train_loss"] for row in history],
        [row["val_loss"] for row in history],
        title="STgram-MFN Training Loss",
        save_path=str(run_dir / "training_loss.png"),
    )

    checkpoint = load_torch_checkpoint(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    full_loader = create_loader(dataset, batch_size=batch_size, shuffle=False, workers=workers)
    features_by_section = extract_features_by_section(model, full_loader, device)
    gmm_components = int(config.get("stgram", {}).get("gmm_components", 2))
    gmm_reg_covar = float(config.get("stgram", {}).get("gmm_reg_covar", 1e-3))
    gmms = {}
    for label, features_for_label in features_by_section.items():
        n_components = min(gmm_components, len(features_for_label))
        gmms[label] = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=gmm_reg_covar,
            random_state=int(config["seed"]),
        ).fit(features_for_label)
        logger.info("Fitted GMM for section label %s with %s samples", label, len(features_for_label))

    gmm_path = run_dir / "gmm_per_section.joblib"
    joblib.dump(gmms, gmm_path)
    save_json(
        {
            "run_dir": str(run_dir),
            "best_model": str(best_path),
            "final_model": str(final_path),
            "gmm_per_section": str(gmm_path),
            "section_to_label": section_to_label,
            "best_val_loss": best_val_loss,
            "epochs": epochs,
        },
        run_dir / "training_summary.json",
    )
    logger.info("STgram-MFN training complete: %s", run_dir)


if __name__ == "__main__":
    main()
