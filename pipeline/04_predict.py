"""Predict anomaly severity for one WAV file using a saved Keras model.

Usage:
    python pipeline/04_predict.py --file Data/gearbox/source_test/sample.wav
    python pipeline/04_predict.py --file sample.wav --model-path artifacts/models/v2/best_model.keras
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.anomaly_scorer import ThresholdResult  # noqa: E402
from src.inference.predictor import (  # noqa: E402
    AudioPredictor,
    fit_reference_threshold,
    load_project_model,
    resolve_latest_model_path,
)
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="04_predict: Predict one audio file")
    parser.add_argument("--file", required=True, dest="audio_file", help="Path to input .wav file.")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path.")
    parser.add_argument("--base-config", default="config/default.yaml", help="Base config YAML path.")
    parser.add_argument("--model-path", default=None, help="Saved .keras model. Defaults to latest artifact model.")
    parser.add_argument("--reference-dir", default=None, help="Normal processed split used to fit threshold.")
    parser.add_argument("--threshold", type=float, default=None, help="Use a supplied fixed anomaly threshold.")
    parser.add_argument("--output", default=None, help="JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.config)
    set_seed(int(config["seed"]))
    logger = get_logger("predict", log_dir=config["artifacts"]["logs_dir"])

    model_path = Path(args.model_path) if args.model_path else resolve_latest_model_path(config["artifacts"]["models_dir"])
    model = load_project_model(model_path)
    threshold = (
        ThresholdResult(value=float(args.threshold), method="fixed")
        if args.threshold is not None
        else fit_reference_threshold(model, config, reference_dir=args.reference_dir)
    )

    result = AudioPredictor(model, config, threshold, model_path).predict_file(args.audio_file)
    output_path = Path(args.output) if args.output else Path("artifacts/predictions") / f"{Path(args.audio_file).stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)

    logger.info("Prediction: %s", result.to_dict())
    logger.info("Prediction JSON: %s", output_path)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
