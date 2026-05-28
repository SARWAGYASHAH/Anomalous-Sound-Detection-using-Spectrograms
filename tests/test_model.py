import numpy as np
import tensorflow as tf
import torch

from src.models import Conv2DAutoencoder, build_autoencoder
from src.models.stgram_mfn import STgramMFN, build_stgram_mfn


def test_autoencoder_forward_pass_keeps_project_input_shape():
    model = build_autoencoder(
        {
            "latent_dim": 4,
            "encoder_layers": [8, 4],
            "decoder_layers": [4, 8],
            "dropout": 0.0,
        },
        input_shape=(128, 313, 1),
    )

    reconstructed = model(tf.zeros((1, 128, 313, 1)), training=False)

    assert tuple(reconstructed.shape) == (1, 128, 313, 1)


def test_autoencoder_serialization_supports_extended_dropout_config(tmp_path):
    model = Conv2DAutoencoder(
        input_shape=(16, 17, 1),
        latent_dim=2,
        encoder_layers=[4],
        decoder_layers=[4],
        dropout=0.0,
        encoder_dropout=0.0,
        decoder_dropout=0.1,
        output_activation=None,
    )
    _ = model(tf.ones((1, 16, 17, 1)), training=False)
    path = tmp_path / "model.keras"
    model.save(path)

    restored = tf.keras.models.load_model(path, compile=False)
    result = restored(np.ones((1, 16, 17, 1), dtype=np.float32), training=False)

    assert tuple(result.shape) == (1, 16, 17, 1)
    assert restored.decoder_dropout == 0.1
    assert restored.output_activation is None


def test_stgram_arcface_metric_logits_do_not_apply_training_margin():
    model = STgramMFN(num_classes=2, embedding_dim=2, use_arcface=True, arcface_scale=1.0)
    with torch.no_grad():
        model.arcface.weight.copy_(torch.eye(2))

    features = torch.tensor([[0.8, 0.6]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)

    metric_logits = model.classification_logits(features)
    margin_logits = model.arcface(features, labels)

    assert metric_logits.argmax(dim=1).item() == 0
    assert margin_logits.argmax(dim=1).item() == 1


def test_stgram_builder_applies_embedding_dropout_config():
    model = build_stgram_mfn({"embedding_dropout": 0.2}, num_classes=3)

    assert model.embedding_dropout.p == 0.2
