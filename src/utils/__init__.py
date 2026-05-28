"""Utility helpers with lazy imports."""

__all__ = ["get_logger", "set_seed"]


def __getattr__(name: str):
    if name == "get_logger":
        from src.utils.logger import get_logger

        return get_logger
    if name == "set_seed":
        from src.utils.seed import set_seed

        return set_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
