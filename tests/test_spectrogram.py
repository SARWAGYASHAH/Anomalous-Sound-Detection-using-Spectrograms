import numpy as np

from src.data.dataset import create_labeled_dataset
from src.data.spectrogram import SpectrogramExtractor


def test_mel_spectrogram_shape_matches_model_input():
    extractor = SpectrogramExtractor(n_fft=1024, hop_length=512, n_mels=128, normalize=True)
    waveform = np.zeros(16000 * 10, dtype=np.float32)

    spectrogram = extractor.extract(waveform, sr=16000, spec_type="mel")

    assert spectrogram.shape == (128, 313)
    assert spectrogram.dtype == np.float32


def test_labeled_dataset_produces_keras_batch_shape(tmp_path):
    for label in ("normal", "anomaly"):
        label_dir = tmp_path / label
        label_dir.mkdir()
        np.save(label_dir / f"{label}.npy", np.ones((128, 313), dtype=np.float32))

    dataset = create_labeled_dataset(tmp_path, batch_size=2, input_shape=(128, 313))
    inputs, labels = next(iter(dataset))

    assert tuple(inputs.shape) == (2, 128, 313, 1)
    assert sorted(labels.numpy().tolist()) == [0, 1]
