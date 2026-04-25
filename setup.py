"""
setup.py - Makes src/ importable as a Python package.

Install in development mode:
    pip install -e .

After installation, you can import any module like:
    from src.utils.logger import get_logger
    from src.training.trainer import Trainer
    from src.data.spectrogram import SpectrogramExtractor
"""

from setuptools import setup, find_packages

setup(
    name="anomalous-sound-detection",
    version="0.1.0",
    description="Anomalous Sound Detection using Spectrograms — Pseudo-MLOps Pipeline",
    author="Sarwagya Shah",
    packages=find_packages(),  # Automatically discovers src/ and all sub-packages
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "mlflow>=2.10.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
