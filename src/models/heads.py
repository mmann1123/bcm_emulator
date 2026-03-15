"""Output heads for PET, PCK, and AET."""

import torch
import torch.nn as nn


class PointwiseHead(nn.Module):
    """1x1 convolution head producing a single output channel.

    Parameters
    ----------
    in_channels : int
        Number of input channels from backbone.
    """

    def __init__(self, in_channels: int, activation: str = None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, T).

        Returns
        -------
        torch.Tensor
            Shape (B, 1, T).
        """
        return self.conv(x)


class AETHead(nn.Module):
    """Unconstrained AET head conditioned on backbone features, PET, and PCK.

    AET <= PET is enforced at inference time after denormalization, not here,
    because the sigmoid*PET constraint is incorrect in z-score normalized space.

    Parameters
    ----------
    in_channels : int
        Number of input channels (backbone_channels + 2 for PET and PCK).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, T) -- backbone features concatenated with PET and PCK.
        pet : torch.Tensor
            Shape (B, 1, T) -- unused, kept for API compatibility.

        Returns
        -------
        torch.Tensor
            Shape (B, 1, T) -- AET prediction in normalized space.
        """
        return self.net(x)
