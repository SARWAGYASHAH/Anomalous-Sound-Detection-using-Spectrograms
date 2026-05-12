import tensorflow as tf

from src.models.autoencoder import build_autoencoder
from src.training.losses import combined_mse_ssim_loss, get_loss


def test_autoencoder_reconstructs_exact_input_shape():
    model = build_autoencoder(
        {
            "input_dim": 128,
            "input_time_frames": 320,
            "latent_dim": 32,
            "encoder_layers": [128, 64, 32],
            "decoder_layers": [32, 64, 128],
            "encoder_dropout": 0.1,
            "decoder_dropout": 0.2,
            "output_activation": "sigmoid",
        }
    )

    batch = tf.zeros((2, 128, 320, 1), dtype=tf.float32)
    reconstructed = model(batch, training=False)

    assert reconstructed.shape == batch.shape
    assert model.latent_dim == 32


def test_combined_loss_is_registered():
    assert get_loss("combined_mse_ssim") is combined_mse_ssim_loss

    y_true = tf.zeros((1, 128, 320, 1), dtype=tf.float32)
    y_pred = tf.zeros((1, 128, 320, 1), dtype=tf.float32)

    assert float(combined_mse_ssim_loss(y_true, y_pred).numpy()) == 0.0
