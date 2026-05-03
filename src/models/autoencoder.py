"""
autoencoder.py - Keras Conv2D autoencoder for spectrogram reconstruction.

The model learns to reconstruct normal mel spectrograms. During inference,
larger reconstruction error becomes the anomaly signal.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from src.models.base_model import BaseModel


def _activation_layer(name: str) -> tf.keras.layers.Layer:
    """Create an activation layer from the config string."""
    normalized = name.lower()
    if normalized == "relu":
        return tf.keras.layers.ReLU()
    if normalized == "leaky_relu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.2)
    if normalized == "elu":
        return tf.keras.layers.ELU()
    raise ValueError(f"Unsupported activation: {name}")


@tf.keras.utils.register_keras_serializable(package="anomalous_sound_detection")
class Conv2DAutoencoder(BaseModel):
    """
    Convolutional autoencoder for normalized spectrogram inputs.

    Args:
        input_shape: Spectrogram shape as (n_mels, time_frames, channels).
        latent_dim: Number of filters in the bottleneck convolution.
        encoder_layers: Conv filter sizes used while downsampling.
        decoder_layers: Conv filter sizes used while upsampling.
        activation: Activation name: relu, leaky_relu, or elu.
        dropout: Dropout rate applied after hidden convolution blocks.
        config: Optional project config snapshot.
        name: Optional Keras model name.
    """

    def __init__(
        self,
        input_shape: tuple[int | None, int | None, int] = (128, None, 1),
        latent_dim: int = 32,
        encoder_layers: list[int] | tuple[int, ...] = (128, 64, 32),
        decoder_layers: list[int] | tuple[int, ...] = (32, 64, 128),
        activation: str = "relu",
        dropout: float = 0.2,
        config: dict[str, Any] | None = None,
        name: str | None = "conv2d_autoencoder",
    ):
        super().__init__(config=config, name=name)
        self.input_spec_shape = tuple(input_shape)
        self.latent_dim = int(latent_dim)
        self.encoder_layers = tuple(int(filters) for filters in encoder_layers)
        self.decoder_layers = tuple(int(filters) for filters in decoder_layers)
        self.activation = activation
        self.dropout = float(dropout)

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.output_layer = tf.keras.layers.Conv2D(
            filters=self.input_spec_shape[-1],
            kernel_size=3,
            padding="same",
            activation="sigmoid",
            name="reconstruction",
        )

    def _build_encoder(self) -> tf.keras.Sequential:
        layers: list[tf.keras.layers.Layer] = []

        for index, filters in enumerate(self.encoder_layers):
            layers.extend(
                [
                    tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=3,
                        strides=2,
                        padding="same",
                        use_bias=False,
                        name=f"encoder_conv_{index + 1}",
                    ),
                    tf.keras.layers.BatchNormalization(name=f"encoder_bn_{index + 1}"),
                    _activation_layer(self.activation),
                ]
            )

            if self.dropout > 0:
                layers.append(tf.keras.layers.Dropout(self.dropout, name=f"encoder_dropout_{index + 1}"))

        layers.extend(
            [
                tf.keras.layers.Conv2D(
                    filters=self.latent_dim,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                    name="bottleneck_conv",
                ),
                tf.keras.layers.BatchNormalization(name="bottleneck_bn"),
                _activation_layer(self.activation),
            ]
        )

        return tf.keras.Sequential(layers, name="encoder")

    def _build_decoder(self) -> tf.keras.Sequential:
        layers: list[tf.keras.layers.Layer] = []

        for index, filters in enumerate(self.decoder_layers):
            layers.extend(
                [
                    tf.keras.layers.UpSampling2D(size=(2, 2), name=f"decoder_upsample_{index + 1}"),
                    tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=3,
                        padding="same",
                        use_bias=False,
                        name=f"decoder_conv_{index + 1}",
                    ),
                    tf.keras.layers.BatchNormalization(name=f"decoder_bn_{index + 1}"),
                    _activation_layer(self.activation),
                ]
            )

            if self.dropout > 0:
                layers.append(tf.keras.layers.Dropout(self.dropout, name=f"decoder_dropout_{index + 1}"))

        return tf.keras.Sequential(layers, name="decoder")

    def encode(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode spectrograms into bottleneck feature maps."""
        return self.encoder(inputs, training=training)

    def decode(
        self,
        encoded: tf.Tensor,
        output_size: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Decode bottleneck feature maps and resize to the original input size."""
        decoded = self.decoder(encoded, training=training)
        decoded = tf.image.resize(decoded, size=output_size, method="bilinear")
        return self.output_layer(decoded, training=training)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Reconstruct a batch of spectrograms with the same shape as inputs."""
        output_size = tf.shape(inputs)[1:3]
        encoded = self.encode(inputs, training=training)
        return self.decode(encoded, output_size=output_size, training=training)

    def get_config(self) -> dict[str, Any]:
        """Return serializable Keras config."""
        return {
            **super().get_config(),
            "input_shape": self.input_spec_shape,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "activation": self.activation,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Conv2DAutoencoder":
        """Recreate a Conv2DAutoencoder from Keras serialization config."""
        config = dict(config)
        project_config = config.pop("project_config", None)
        name = config.pop("name", None)
        config.pop("trainable", None)
        config.pop("dtype", None)
        return cls(config=project_config, name=name, **config)


def build_autoencoder(
    model_config: dict[str, Any],
    input_shape: tuple[int | None, int | None, int] | None = None,
) -> Conv2DAutoencoder:
    """
    Build a Conv2DAutoencoder from the project model config.

    Args:
        model_config: Usually config["model"]["autoencoder"].
        input_shape: Optional explicit channels-last input shape.

    Returns:
        Uncompiled Conv2DAutoencoder instance.
    """
    cfg = dict(model_config)

    if input_shape is None:
        input_dim = cfg.get("input_dim", 128)
        input_shape = (input_dim, None, 1)

    return Conv2DAutoencoder(
        input_shape=input_shape,
        latent_dim=cfg.get("latent_dim", 32),
        encoder_layers=cfg.get("encoder_layers", (128, 64, 32)),
        decoder_layers=cfg.get("decoder_layers", (32, 64, 128)),
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.2),
        config=cfg,
    )
