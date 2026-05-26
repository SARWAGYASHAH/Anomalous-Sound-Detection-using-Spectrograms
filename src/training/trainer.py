"""
trainer.py - Keras training utilities.

The trainer is designed for Google Colab/full GPU runs. Local use should stay
limited to imports, model shape checks, and tiny dry checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tensorflow as tf

from src.training.losses import get_loss
from src.utils.artifact_versioner import ArtifactVersioner
from src.utils.logger import get_logger
from src.utils.metadata_tracker import MetadataTracker
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TrainingArtifacts:
    """Paths produced by a training run."""

    run_id: str
    version_dir: Path
    model_path: Path
    best_model_path: Path
    final_model_path: Path
    config_snapshot_path: Path
    metadata_path: Path | None = None


class KerasTrainer:
    """
    Compile, train, and save a Keras autoencoder with project conventions.

    Args:
        model: Uncompiled Keras model.
        config: Full project config dictionary.
        version_dir: Optional explicit output directory. When omitted, the
            configured versioning policy resolves one.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: dict[str, Any],
        version_dir: str | Path | None = None,
    ):
        self.model = model
        self.config = config
        self.training_config = config.get("training", {})
        self.mlflow_config = config.get("mlflow", {})

        set_seed(config.get("seed", 42))

        versioner = ArtifactVersioner(config.get("versioning", {}).get("base_dir", "artifacts/models"))
        self.version_dir = Path(version_dir) if version_dir is not None else versioner.resolve_version(config)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.config_snapshot_path = versioner.save_config_snapshot(self.version_dir, config)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = self.version_dir / "model.keras"
        self.best_model_path = self.version_dir / "best_model.keras"
        self.final_model_path = self.version_dir / "final_model.keras"
        self.history: tf.keras.callbacks.History | None = None

    def compile(self) -> tf.keras.Model:
        """Compile the model from the training config."""
        optimizer = self._build_optimizer()
        loss = get_loss(self.training_config.get("loss", "mse"))

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
        )

        logger.info(
            "Compiled model with optimizer=%s, loss=%s",
            self.training_config.get("optimizer", "adam"),
            self.training_config.get("loss", "mse"),
        )
        return self.model

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset | None = None,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
    ) -> tuple[tf.keras.callbacks.History, TrainingArtifacts]:
        """
        Train the model and save final artifacts.

        This is the method intended for Colab execution.
        """
        if getattr(self.model, "optimizer", None) is None:
            self.compile()

        callbacks = self._build_callbacks(has_validation=val_dataset is not None) + list(callbacks or [])
        epochs = int(self.training_config.get("epochs", 50))

        logger.info("Training started for %s epochs. Version dir: %s", epochs, self.version_dir)

        if self.mlflow_config.get("enabled", False):
            history = self._fit_with_mlflow(train_dataset, val_dataset, callbacks, epochs)
        else:
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
            )

        self.history = history
        self.model.save(self.final_model_path)
        self.model.save(self.model_path)

        artifacts = TrainingArtifacts(
            run_id=self.run_id,
            version_dir=self.version_dir,
            model_path=self.model_path,
            best_model_path=self.best_model_path,
            final_model_path=self.final_model_path,
            config_snapshot_path=self.config_snapshot_path,
        )

        artifacts.metadata_path = self._save_metadata(history, artifacts)
        logger.info("Training complete. Final model: %s", self.final_model_path)
        return history, artifacts

    def _build_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer_name = self.training_config.get("optimizer", "adam").lower()
        learning_rate = float(self.training_config.get("learning_rate", 0.001))
        weight_decay = float(self.training_config.get("weight_decay", 0.0))

        if optimizer_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if optimizer_name == "adamw":
            return tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        if optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _build_callbacks(self, has_validation: bool = True) -> list[tf.keras.callbacks.Callback]:
        monitor = "val_loss" if has_validation else "loss"

        callbacks: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.best_model_path),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(self.version_dir / "training_log.csv")),
        ]

        early_stopping_config = self.training_config.get("early_stopping", {})
        if early_stopping_config.get("enabled", True):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=int(early_stopping_config.get("patience", 10)),
                    min_delta=float(early_stopping_config.get("min_delta", 0.0001)),
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        scheduler = self._build_lr_scheduler()
        if scheduler is not None:
            callbacks.append(scheduler)

        return callbacks

    def _build_lr_scheduler(self) -> tf.keras.callbacks.Callback | None:
        scheduler_config = self.training_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "none").lower()
        initial_lr = float(self.training_config.get("learning_rate", 0.001))

        if scheduler_type == "none":
            return None

        if scheduler_type == "step":
            step_size = int(scheduler_config.get("step_size", 10))
            gamma = float(scheduler_config.get("gamma", 0.5))

            def step_schedule(epoch: int, lr: float) -> float:
                if epoch > 0 and epoch % step_size == 0:
                    return float(lr * gamma)
                return float(lr)

            return tf.keras.callbacks.LearningRateScheduler(step_schedule, verbose=1)

        if scheduler_type == "cosine":
            epochs = int(self.training_config.get("epochs", 50))

            def cosine_schedule(epoch: int, _: float) -> float:
                progress = epoch / max(epochs - 1, 1)
                cosine_decay = 0.5 * (1.0 + tf.math.cos(tf.constant(progress * 3.141592653589793)))
                return float(initial_lr * cosine_decay.numpy())

            return tf.keras.callbacks.LearningRateScheduler(cosine_schedule, verbose=1)

        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _fit_with_mlflow(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset | None,
        callbacks: list[tf.keras.callbacks.Callback],
        epochs: int,
    ) -> tf.keras.callbacks.History:
        try:
            import mlflow
            import mlflow.tensorflow
        except ImportError as exc:
            logger.warning("MLflow is enabled but unavailable. Training without MLflow: %s", exc)
            return self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
            )

        mlflow.set_tracking_uri(self.mlflow_config.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(self.mlflow_config.get("experiment_name", "gearbox_anomaly_detection"))

        with mlflow.start_run(run_name=self.run_id) as run:
            mlflow.set_tags(self.mlflow_config.get("tags", {}))
            mlflow.log_params(self._metadata_params())

            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
            )

            for metric_name, values in history.history.items():
                if values:
                    mlflow.log_metric(f"final_{metric_name}", float(values[-1]))

            mlflow.log_artifact(str(self.config_snapshot_path))
            if self.mlflow_config.get("log_models", True):
                mlflow.tensorflow.log_model(self.model, artifact_path="model")

            self.mlflow_run_id = run.info.run_id

        return history

    def _save_metadata(
        self,
        history: tf.keras.callbacks.History,
        artifacts: TrainingArtifacts,
    ) -> Path:
        tracker = MetadataTracker(self.config.get("artifacts", {}).get("metadata_dir", "artifacts/metadata"))

        metrics = {
            f"final_{name}": float(values[-1])
            for name, values in history.history.items()
            if values
        }

        return tracker.save(
            run_id=self.run_id,
            params=self._metadata_params(),
            metrics=metrics,
            artifacts={
                "version_dir": str(artifacts.version_dir),
                "model_path": str(artifacts.model_path),
                "best_model_path": str(artifacts.best_model_path),
                "final_model_path": str(artifacts.final_model_path),
                "config_snapshot_path": str(artifacts.config_snapshot_path),
            },
            mlflow_run_id=getattr(self, "mlflow_run_id", None),
            experiment_name=self.mlflow_config.get("experiment_name"),
        )

    def _metadata_params(self) -> dict[str, Any]:
        model_config = self.config.get("model", {}).get("autoencoder", {})
        return {
            "seed": self.config.get("seed"),
            "epochs": self.training_config.get("epochs"),
            "batch_size": self.training_config.get("batch_size"),
            "learning_rate": self.training_config.get("learning_rate"),
            "optimizer": self.training_config.get("optimizer"),
            "loss": self.training_config.get("loss"),
            "latent_dim": model_config.get("latent_dim"),
            "dropout": model_config.get("dropout"),
            "n_mels": self.config.get("spectrogram", {}).get("n_mels"),
        }
