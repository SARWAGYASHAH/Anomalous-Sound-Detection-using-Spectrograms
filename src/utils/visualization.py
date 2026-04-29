"""
visualization.py - Plotting utilities for spectrograms, anomaly scores, and training.

All plots can be saved to disk and/or displayed interactively.

Usage:
    from src.utils.visualization import plot_spectrogram, plot_loss_curve, plot_anomaly_scores

    plot_spectrogram(spectrogram_array, title="Normal Gearbox", save_path="plots/spec.png")
    plot_loss_curve(train_losses, val_losses, save_path="plots/loss.png")
    plot_anomaly_scores(scores, labels, save_path="plots/scores.png")
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------- Global Style ----------
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    "figure.figsize": (12, 4),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_spectrogram(
    spectrogram: np.ndarray,
    sr: int = 16000,
    hop_length: int = 512,
    title: str = "Mel Spectrogram",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot a single spectrogram as a heatmap.

    Args:
        spectrogram: 2D numpy array (n_mels × time_frames).
        sr: Sample rate (used for axis labeling).
        hop_length: Hop length (used for time axis calculation).
        title: Plot title.
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    img = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Bands")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Power (dB)")

    plt.tight_layout()
    _save_and_show(fig, save_path, show)


def plot_spectrogram_comparison(
    spec_normal: np.ndarray,
    spec_anomalous: np.ndarray,
    title_normal: str = "Normal",
    title_anomalous: str = "Anomalous",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot two spectrograms side by side for comparison.

    Args:
        spec_normal: Spectrogram from a normal sample.
        spec_anomalous: Spectrogram from an anomalous sample.
        title_normal: Title for the normal spectrogram.
        title_anomalous: Title for the anomalous spectrogram.
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    for ax, spec, title in zip(axes, [spec_normal, spec_anomalous],
                                [title_normal, title_anomalous]):
        img = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Bands")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    _save_and_show(fig, save_path, show)


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Training Loss Curve",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot training and optional validation loss curves.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses: Optional list of validation loss values per epoch.
        title: Plot title.
        save_path: If provided, save the figure.
        show: If True, display the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss", alpha=0.9)

    if val_losses is not None:
        ax.plot(epochs, val_losses, "r--", linewidth=2, label="Val Loss", alpha=0.9)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_show(fig, save_path, show)


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray | None = None,
    threshold: float | None = None,
    title: str = "Anomaly Score Distribution",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot anomaly score distribution with optional ground truth coloring.

    Args:
        scores: 1D array of anomaly scores.
        labels: Optional 1D array of ground truth labels (0=normal, 1=anomaly).
        threshold: If provided, draw a vertical threshold line.
        title: Plot title.
        save_path: If provided, save the figure.
        show: If True, display the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if labels is not None:
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        ax.hist(normal_scores, bins=50, alpha=0.6, color="#2196F3",
                label=f"Normal (n={len(normal_scores)})", density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.6, color="#F44336",
                label=f"Anomaly (n={len(anomaly_scores)})", density=True)
    else:
        ax.hist(scores, bins=50, alpha=0.7, color="#9C27B0",
                label="All Scores", density=True)

    if threshold is not None:
        ax.axvline(x=threshold, color="#FF9800", linewidth=2.5,
                   linestyle="--", label=f"Threshold = {threshold:.4f}")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    _save_and_show(fig, save_path, show)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr: False positive rates from sklearn.metrics.roc_curve.
        tpr: True positive rates from sklearn.metrics.roc_curve.
        auc_score: Pre-computed AUC score for legend.
        title: Plot title.
        save_path: If provided, save the figure.
        show: If True, display the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.plot(fpr, tpr, "b-", linewidth=2.5, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_show(fig, save_path, show)


def plot_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Reconstruction Comparison",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot original vs reconstructed spectrogram and their error map.

    Args:
        original: Original spectrogram (2D numpy array).
        reconstructed: Reconstructed spectrogram from autoencoder.
        title: Plot title.
        save_path: If provided, save the figure.
        show: If True, display the plot.
    """
    error = np.abs(original - reconstructed)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    titles = ["Original", "Reconstructed", "Error Map"]
    data = [original, reconstructed, error]
    cmaps = ["magma", "magma", "hot"]

    for ax, d, t, cmap in zip(axes, data, titles, cmaps):
        img = ax.imshow(d, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(t, fontweight="bold")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Bands")
        fig.colorbar(img, ax=ax)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_and_show(fig, save_path, show)


# ---------- Internal Helper ----------

def _save_and_show(fig, save_path: str | None, show: bool) -> None:
    """Save figure to disk and/or display it."""
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
