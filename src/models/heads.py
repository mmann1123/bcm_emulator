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
    """Stress-fraction AET head with multiplicative inductive bias.

    Mirrors BCMv8: AET = Kv × PET × f(soil_water/AWC).
    - stress_net learns f(soil_water) ∈ [0, 1] via sigmoid
    - Multiplicative: stress × Kv × PET (in normalized space)
    - correction_net handles z-score offset + Kv=0 fallback

    Parameters
    ----------
    bb_channels : int
        Backbone output channels (256).
    hidden_dim : int
        Hidden size for sub-networks.
    dropout : float
        Dropout rate.
    """

    def __init__(self, bb_channels: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        # backbone(256) + kbdi(1) + pck(1) = 258 → stress ∈ [0,1]
        self.stress_net = nn.Sequential(
            nn.Conv1d(bb_channels + 2, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        # backbone(256) + kbdi(1) + pet(1) + pck(1) = 259 → residual
        self.correction_net = nn.Sequential(
            nn.Conv1d(bb_channels + 3, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, features, pet, pck, kbdi, kv):
        """
        features: (B, 256, T), pet/pck/kbdi/kv: (B, 1, T)
        Returns: (B, 1, T) AET in z-score normalized space.
        """
        stress = torch.sigmoid(self.stress_net(
            torch.cat([features, kbdi, pck], dim=1)
        ))  # (B, 1, T) ∈ [0, 1]

        # Clamp stress*kv ≤ 1.0 so mult path can't produce AET > PET.
        # Without this, Kv=1.517 (redwoods) × stress≈1.0 → AET ≈ 1.5×PET
        # in normalized space. The post-denorm clamp in evaluate.py handles
        # inference, but during training the loss would see unclamped values
        # and receive physically incorrect gradients.
        aet_frac = torch.clamp(stress * kv, max=1.0)  # (B, 1, T)
        mult = aet_frac * pet  # multiplicative pathway

        correction = self.correction_net(
            torch.cat([features, kbdi, pet, pck], dim=1)
        )  # normalization offset + Kv=0 fallback

        return mult + correction
