"""Project model definitions with lazy backend imports."""

__all__ = [
    "BaseModel",
    "Conv2DAutoencoder",
    "STgramMFN",
    "build_autoencoder",
    "build_stgram_mfn",
    "freeze_feature_extractor",
]


def __getattr__(name: str):
    if name in {"BaseModel"}:
        from src.models.base_model import BaseModel

        return BaseModel
    if name in {"Conv2DAutoencoder", "build_autoencoder"}:
        from src.models.autoencoder import Conv2DAutoencoder, build_autoencoder

        return {"Conv2DAutoencoder": Conv2DAutoencoder, "build_autoencoder": build_autoencoder}[name]
    if name in {"STgramMFN", "build_stgram_mfn", "freeze_feature_extractor"}:
        from src.models.stgram_mfn import STgramMFN, build_stgram_mfn, freeze_feature_extractor

        return {
            "STgramMFN": STgramMFN,
            "build_stgram_mfn": build_stgram_mfn,
            "freeze_feature_extractor": freeze_feature_extractor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
