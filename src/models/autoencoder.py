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
        input_shape: tuple[int | None, int | None, int] = (128, 320, 1),
        latent_dim: int = 32,
        encoder_layers: list[int] | tuple[int, ...] = (128, 64, 32),
        decoder_layers: list[int] | tuple[int, ...] = (32, 64, 128),
        activation: str = "relu",
        dropout: float = 0.2,
        encoder_dropout: float | None = None,
        decoder_dropout: float | None = None,
        output_activation: str | None = "sigmoid",
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
        self.encoder_dropout = float(dropout if encoder_dropout is None else encoder_dropout)
        self.decoder_dropout = float(dropout if decoder_dropout is None else decoder_dropout)
        self.output_activation = output_activation

        self.encoder = self._build_encoder()
        self.bottleneck: tf.keras.Sequential | None = None
        self.decoder = self._build_decoder()
        self.output_layer = tf.keras.layers.Conv2D(
            filters=self.input_spec_shape[-1],
            kernel_size=3,
            padding="same",
            activation=self.output_activation,
            name="reconstruction",
        )
        if self.input_spec_shape[0] is not None and self.input_spec_shape[1] is not None:
            self._init_bottleneck_layers(self._encoded_shape_for(self.input_spec_shape))

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

            if self.encoder_dropout > 0:
                layers.append(tf.keras.layers.Dropout(self.encoder_dropout, name=f"encoder_dropout_{index + 1}"))

        return tf.keras.Sequential(layers, name="encoder")

    def _encoded_shape_for(
        self,
        input_shape: tuple[int | None, int | None, int] | tf.TensorShape,
    ) -> tuple[int, int, int]:
        """Compute the encoder feature-map shape after stride-2 same convolutions."""
        shape = tuple(tf.TensorShape(input_shape).as_list())
        height, width = shape[0], shape[1]

        if height is None or width is None:
            raise ValueError(
                "Dense bottleneck requires fixed spectrogram height and width. "
                "Pass input_shape=(n_mels, time_frames, channels), e.g. (128, 320, 1)."
            )

        for _ in self.encoder_layers:
            height = (int(height) + 1) // 2
            width = (int(width) + 1) // 2

        return int(height), int(width), int(self.encoder_layers[-1])

    def _init_bottleneck_layers(self, encoded_shape: tuple[int, int, int]) -> None:
        """Create the Flatten -> Dense(latent) -> Dense -> Reshape bottleneck."""
        if self.bottleneck is not None:
            return

        flattened_units = int(encoded_shape[0] * encoded_shape[1] * encoded_shape[2])
        self.bottleneck = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(name="bottleneck_flatten"),
                tf.keras.layers.Dense(self.latent_dim, name="latent_vector"),
                _activation_layer(self.activation),
                tf.keras.layers.Dense(flattened_units, name="bottleneck_expand"),
                _activation_layer(self.activation),
                tf.keras.layers.Reshape(encoded_shape, name="bottleneck_reshape"),
            ],
            name="dense_bottleneck",
        )

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

            if self.decoder_dropout > 0:
                layers.append(tf.keras.layers.Dropout(self.decoder_dropout, name=f"decoder_dropout_{index + 1}"))

        return tf.keras.Sequential(layers, name="decoder")

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layers that depend on the static spectrogram width."""
        self._init_bottleneck_layers(self._encoded_shape_for(input_shape[1:]))
        super().build(input_shape)

    def encode(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode spectrograms into bottleneck feature maps."""
        encoded = self.encoder(inputs, training=training)
        if self.bottleneck is None:
            self._init_bottleneck_layers(self._encoded_shape_for(inputs.shape[1:]))
        return self.bottleneck(encoded, training=training)

    def decode(
        self,
        encoded: tf.Tensor,
        output_size: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Decode bottleneck feature maps and resize to the original input size."""
        decoded = self.decoder(encoded, training=training)
        reconstruction = self.output_layer(decoded, training=training)
        reconstruction = tf.image.resize_with_crop_or_pad(
            reconstruction,
            target_height=output_size[0],
            target_width=output_size[1],
        )
        return reconstruction

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Reconstruct a batch of spectrograms with the same shape as inputs."""
        output_size = tf.shape(inputs)[1:3]
        encoded = self.encode(inputs, training=training)
        reconstruction = self.decode(encoded, output_size=output_size, training=training)
        shape_check = tf.debugging.assert_equal(
            tf.shape(reconstruction)[1:3],
            tf.shape(inputs)[1:3],
            message="Autoencoder output spatial shape must match input spatial shape.",
        )
        with tf.control_dependencies([shape_check]):
            return tf.identity(reconstruction)

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
            "encoder_dropout": self.encoder_dropout,
            "decoder_dropout": self.decoder_dropout,
            "output_activation": self.output_activation,
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
        input_time_frames = cfg.get("input_time_frames", 320)
        input_shape = (input_dim, input_time_frames, 1)

    return Conv2DAutoencoder(
        input_shape=input_shape,
        latent_dim=cfg.get("latent_dim", 32),
        encoder_layers=cfg.get("encoder_layers", (128, 64, 32)),
        decoder_layers=cfg.get("decoder_layers", (32, 64, 128)),
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.2),
        encoder_dropout=cfg.get("encoder_dropout"),
        decoder_dropout=cfg.get("decoder_dropout"),
        output_activation=cfg.get("output_activation", "sigmoid"),
        config=cfg,
    )
