"""Data loading helpers with lazy backend imports."""

__all__ = [
    "AudioLoader",
    "SpectrogramExtractor",
    "LABEL_MAP",
    "STgramWaveDataset",
    "build_section_label_map",
    "create_autoencoder_dataset",
    "create_autoencoder_dataset_from_files",
    "create_autoencoder_train_val_datasets",
    "create_labeled_dataset",
    "discover_spectrogram_files",
    "discover_wav_files",
    "extract_binary_label",
    "extract_section",
    "get_labels",
    "load_spectrogram_array",
]


def __getattr__(name: str):
    if name == "AudioLoader":
        from src.data.audio_loader import AudioLoader

        return AudioLoader
    if name == "SpectrogramExtractor":
        from src.data.spectrogram import SpectrogramExtractor

        return SpectrogramExtractor
    if name in {
        "LABEL_MAP",
        "create_autoencoder_dataset",
        "create_autoencoder_dataset_from_files",
        "create_autoencoder_train_val_datasets",
        "create_labeled_dataset",
        "discover_spectrogram_files",
        "get_labels",
        "load_spectrogram_array",
    }:
        from src.data import dataset

        return getattr(dataset, name)
    if name in {
        "STgramWaveDataset",
        "build_section_label_map",
        "discover_wav_files",
        "extract_binary_label",
        "extract_section",
    }:
        from src.data import stgram_dataset

        return getattr(stgram_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
