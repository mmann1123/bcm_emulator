"""Output heads for PET, PCK, and AET."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseHead(nn.Module):
    """1x1 convolution head producing a single output channel.

    Parameters
    ----------
    in_channels : int
        Number of input channels from backbone.
    activation : str or None
        'softplus' for PCK and PET (non-negative), None for unconstrained.
    """

    def __init__(self, in_channels: int, activation: str = None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.activation = activation

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
        out = self.conv(x)
        if self.activation == "softplus":
            out = F.softplus(out)
        return out


class AETStressHead(nn.Module):
    """AET = sigmoid(f) * PET, guaranteeing AET <= PET.

    Takes backbone features concatenated with PET and PCK predictions.

    Parameters
    ----------
    in_channels : int
        Number of input channels (backbone_channels + 2 for PET and PCK).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, T) -- backbone features concatenated with PET and PCK.
        pet : torch.Tensor
            Shape (B, 1, T) -- PET predictions for AET constraint.

        Returns
        -------
        torch.Tensor
            Shape (B, 1, T) -- AET = sigmoid(f) * PET.
        """
        stress_fraction = torch.sigmoid(self.conv(x))
        aet = stress_fraction * pet
        return aet
