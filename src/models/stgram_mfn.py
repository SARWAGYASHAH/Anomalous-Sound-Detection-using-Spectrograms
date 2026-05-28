"""
PyTorch STgram-MFN model for anomalous sound detection.

The model fuses a raw-waveform temporal gram branch with a log-mel
spectrogram branch, learns section-aware embeddings with ArcFace, and exposes
128-dimensional features for density-based anomaly scoring.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace angular-margin classification head."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class TgramNet(nn.Module):
    """Learned temporal representation from raw waveform samples."""

    def __init__(
        self,
        out_channels: int = 128,
        win_len: int = 1024,
        hop_len: int = 512,
        layers: int = 3,
        num_frames: int = 313,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.win_len = int(win_len)
        self.hop_len = int(hop_len)
        self.num_frames = int(num_frames)
        self.conv_extractor = nn.Conv1d(
            1,
            self.out_channels,
            kernel_size=self.win_len,
            stride=self.hop_len,
            padding=self.win_len // 2,
            bias=False,
        )
        self.conv_encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LayerNorm(self.num_frames),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(
                        self.out_channels,
                        self.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                )
                for _ in range(int(layers))
            ]
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        x = self.conv_extractor(waveform)
        x = _match_time_frames(x, self.num_frames)
        return self.conv_encoder(x)


class ConvBlock(nn.Module):
    """Conv-BN-PReLU block used by the MobileFaceNet backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.PReLU(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Bottleneck(nn.Module):
    """MobileFaceNet inverted residual bottleneck."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int) -> None:
        super().__init__()
        hidden_channels = int(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=1, activation=True),
            ConvBlock(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                activation=True,
            ),
            ConvBlock(hidden_channels, out_channels, kernel_size=1, activation=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        if self.use_residual:
            return inputs + outputs
        return outputs


class MobileFaceNetBackbone(nn.Module):
    """Compact CNN backbone that converts fused grams into embeddings."""

    def __init__(
        self,
        input_channels: int = 2,
        embedding_dim: int = 128,
        base_channels: int = 64,
        bottleneck_channels: int = 128,
        expansion_schedule: tuple[int, ...] = (2, 2, 4, 4, 4, 4),
        stride_schedule: tuple[int, ...] = (2, 1, 2, 1, 2, 1),
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(input_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.dw_conv1 = ConvBlock(
            base_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=base_channels,
        )

        blocks: list[nn.Module] = []
        in_channels = base_channels
        for stride, expansion in zip(stride_schedule, expansion_schedule):
            blocks.append(Bottleneck(in_channels, bottleneck_channels, stride=stride, expansion=expansion))
            in_channels = bottleneck_channels
        self.blocks = nn.Sequential(*blocks)
        self.conv2 = ConvBlock(bottleneck_channels, 512, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inputs)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return self.embedding(x)


class STgramMFN(nn.Module):
    """Spectral-temporal fusion model with optional ArcFace head."""

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 128,
        num_frames: int = 313,
        embedding_dim: int = 128,
        win_len: int = 1024,
        hop_len: int = 512,
        tgram_layers: int = 3,
        use_arcface: bool = True,
        arcface_margin: float = 0.5,
        arcface_scale: float = 30.0,
        base_channels: int = 64,
        bottleneck_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.n_mels = int(n_mels)
        self.num_frames = int(num_frames)
        self.embedding_dim = int(embedding_dim)
        self.use_arcface = bool(use_arcface)

        self.tgramnet = TgramNet(
            out_channels=self.n_mels,
            win_len=win_len,
            hop_len=hop_len,
            layers=tgram_layers,
            num_frames=self.num_frames,
        )
        self.mobilefacenet = MobileFaceNetBackbone(
            input_channels=2,
            embedding_dim=self.embedding_dim,
            base_channels=base_channels,
            bottleneck_channels=bottleneck_channels,
        )
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        self.arcface = ArcMarginProduct(
            self.embedding_dim,
            self.num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )

    def extract_features(self, waveform: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim == 4 and mel.shape[1] == 1:
            mel = mel.squeeze(1)
        if mel.ndim != 3:
            raise ValueError(f"Expected mel shape [B, n_mels, frames], got {tuple(mel.shape)}")

        tgram = self.tgramnet(waveform)
        mel = _match_time_frames(mel, tgram.shape[-1])
        fused = torch.stack([mel, tgram], dim=1)
        features = self.mobilefacenet(fused)
        return F.normalize(features, dim=1)

    def forward(
        self,
        waveform: torch.Tensor,
        mel: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(waveform, mel)
        if self.use_arcface and labels is not None:
            logits = self.arcface(features, labels)
        else:
            logits = self.classifier(features)
        return logits, features


def build_stgram_mfn(model_config: dict, num_classes: int) -> STgramMFN:
    """Build STgramMFN from a YAML model config section."""
    return STgramMFN(
        num_classes=num_classes,
        n_mels=int(model_config.get("n_mels", 128)),
        num_frames=int(model_config.get("num_frames", 313)),
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        win_len=int(model_config.get("win_len", 1024)),
        hop_len=int(model_config.get("hop_len", 512)),
        tgram_layers=int(model_config.get("tgram_layers", 3)),
        use_arcface=bool(model_config.get("use_arcface", True)),
        arcface_margin=float(model_config.get("arcface_margin", 0.5)),
        arcface_scale=float(model_config.get("arcface_scale", 30.0)),
        base_channels=int(model_config.get("base_channels", 64)),
        bottleneck_channels=int(model_config.get("bottleneck_channels", 128)),
    )


def freeze_feature_extractor(model: STgramMFN) -> None:
    """Freeze the fusion backbone while leaving classification heads trainable."""
    for module in (model.tgramnet, model.mobilefacenet):
        for parameter in module.parameters():
            parameter.requires_grad = False


def _match_time_frames(inputs: torch.Tensor, frames: int) -> torch.Tensor:
    """Crop or pad the last dimension to a fixed frame count."""
    current = inputs.shape[-1]
    if current == frames:
        return inputs
    if current > frames:
        return inputs[..., :frames]
    pad = frames - current
    return F.pad(inputs, (0, pad))
