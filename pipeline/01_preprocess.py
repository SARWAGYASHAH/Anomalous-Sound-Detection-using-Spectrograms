"""
01_preprocess.py - Audio → Spectrogram preprocessing pipeline.

Reads raw .wav files from the DCASE gearbox dataset, converts them to
mel spectrograms, and saves as .npy files organized by split and label.

Output structure:
    Data/processed/
    ├── train/
    │   └── normal/
    │       ├── section_00_source_train_normal_0000.npy
    │       └── ...
    ├── source_test/
    │   ├── normal/
    │   │   └── ...
    │   └── anomaly/
    │       └── ...
    └── target_test/
        ├── normal/
        │   └── ...
        └── anomaly/
            └── ...

Usage:
    python pipeline/01_preprocess.py
    python pipeline/01_preprocess.py --config config/experiment_01.yaml
    python pipeline/01_preprocess.py --save-png    # Also save spectrogram images
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.audio_loader import AudioLoader
from src.data.spectrogram import SpectrogramExtractor
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def load_config(
    base_path: str = "config/default.yaml",
    override_path: str | None = None,
) -> dict:
    """Load base config and optionally merge with an override config."""

    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    if override_path:
        with open(override_path, "r") as f:
            override = yaml.safe_load(f)
        config = deep_merge(config, override)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def process_directory(
    input_dir: Path,
    output_dir: Path,
    loader: AudioLoader,
    extractor: SpectrogramExtractor,
    spec_type: str,
    sr: int,
    save_png: bool = False,
    logger=None,
) -> dict:
    """
    Process all .wav files in a directory → save as .npy spectrograms.

    Returns:
        Dict with processing stats.
    """
    files = loader.discover_files(input_dir)

    if not files:
        logger.warning(f"No .wav files found in {input_dir}")
        return {"total": 0, "normal": 0, "anomaly": 0}

    stats = {"total": 0, "normal": 0, "anomaly": 0}

    for i, file_info in enumerate(files):
        label = file_info["label"]
        filename = Path(file_info["filename"]).stem

        # Create label subdirectory
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Load audio
        waveform, _ = loader.load(file_info["filepath"])

        # Extract spectrogram
        spectrogram = extractor.extract(waveform, sr=sr, spec_type=spec_type)

        # Save as .npy (primary format for training)
        npy_path = label_dir / f"{filename}.npy"
        extractor.save_as_numpy(spectrogram, npy_path)

        # Optionally save as .png
        if save_png:
            png_path = label_dir / f"{filename}.png"
            extractor.save_as_image(spectrogram, png_path, sr=sr)

        stats["total"] += 1
        stats[label] = stats.get(label, 0) + 1

        # Progress logging
        if (i + 1) % 200 == 0 or (i + 1) == len(files):
            logger.info(f"  Processed {i + 1}/{len(files)} files")

    return stats


def main():
    """Main preprocessing pipeline."""
    # ---------- CLI Arguments ----------
    parser = argparse.ArgumentParser(description="01_preprocess: Audio → Spectrograms")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to override config YAML (merged on top of default.yaml)",
    )
    parser.add_argument(
        "--save-png", action="store_true",
        help="Also save spectrograms as .png images (slower, for visualization)",
    )
    args = parser.parse_args()

    # ---------- Load Config ----------
    config = load_config(override_path=args.config)

    # ---------- Setup ----------
    set_seed(config["seed"])
    logger = get_logger(
        "preprocess",
        log_dir=config["artifacts"]["logs_dir"],
    )

    logger.info("=" * 60)
    logger.info("PIPELINE STAGE 1: Preprocessing (Audio → Spectrograms)")
    logger.info("=" * 60)
    start_time = time.time()

    # ---------- Initialize Components ----------
    audio_cfg = config["audio"]
    spec_cfg = config["spectrogram"]

    loader = AudioLoader(
        target_sr=audio_cfg["sample_rate"],
        mono=audio_cfg["mono"],
        duration=audio_cfg["duration"],
    )

    extractor = SpectrogramExtractor(
        n_fft=spec_cfg["n_fft"],
        hop_length=spec_cfg["hop_length"],
        n_mels=spec_cfg["n_mels"],
        power=spec_cfg["power"],
        normalize=spec_cfg["normalize"],
    )

    # ---------- Define Splits ----------
    machine_type = config["data"]["machine_type"]
    raw_base = Path(f"Data/{machine_type}")
    processed_base = Path(config["data"]["processed_dir"])

    splits = {
        "train": raw_base / "train",
        "source_test": raw_base / "source_test",
        "target_test": raw_base / "target_test",
    }

    # ---------- Process Each Split ----------
    all_stats = {}

    for split_name, input_dir in splits.items():
        logger.info(f"\nProcessing split: {split_name}")
        logger.info(f"  Input:  {input_dir}")

        output_dir = processed_base / split_name
        logger.info(f"  Output: {output_dir}")

        if not input_dir.exists():
            logger.warning(f"  Skipping — directory not found: {input_dir}")
            continue

        stats = process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            loader=loader,
            extractor=extractor,
            spec_type=spec_cfg["type"],
            sr=audio_cfg["sample_rate"],
            save_png=args.save_png,
            logger=logger,
        )

        all_stats[split_name] = stats
        logger.info(f"  Done: {stats}")

    # ---------- Summary ----------
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Time elapsed: {elapsed:.1f}s")

    for split_name, stats in all_stats.items():
        logger.info(f"  {split_name}: {stats['total']} spectrograms "
                     f"(normal={stats.get('normal', 0)}, anomaly={stats.get('anomaly', 0)})")

    total = sum(s["total"] for s in all_stats.values())
    logger.info(f"  Total: {total} spectrograms saved to {processed_base}")
    logger.info(f"  Format: .npy (shape: {spec_cfg['n_mels']} × time_frames)")

    if args.save_png:
        logger.info("  PNG images also saved for visualization.")


if __name__ == "__main__":
    main()
