"""Causal temporal convolution building blocks."""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding to preserve sequence length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=self.padding, dilation=dilation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove right-side padding to enforce causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual block with two causal dilated convolutions.

    Architecture: CausalConv -> ReLU -> Dropout -> CausalConv -> ReLU -> Dropout + residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 conv for residual connection if channels change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
