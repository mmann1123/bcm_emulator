"""TCN Backbone: 5-level causal dilated temporal convolutional network."""

import torch
import torch.nn as nn

from .layers import TemporalBlock


class TCNBackbone(nn.Module):
    """5-level causal TCN with dilations 1, 2, 4, 8, 16.

    Receptive field = 1 + 2*(k-1)*sum(dilations) = 1 + 2*2*(1+2+4+8+16) = 125 months.

    Parameters
    ----------
    in_channels : int
        Number of input channels (25 for BCM emulator: 10 dyn + 7 static + 8 fveg).
    channels : list of int
        Channel sizes for each level, default [64, 128, 128, 256, 256].
    kernel_size : int
        Kernel size for causal convolutions, default 3.
    dropout : float
        Dropout rate, default 0.1.
    """

    def __init__(
        self,
        in_channels: int = 13,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 128, 256, 256]

        dilations = [1, 2, 4, 8, 16]
        assert len(channels) == len(dilations)

        layers = []
        for i, (c, d) in enumerate(zip(channels, dilations)):
            in_c = in_channels if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_c, c, kernel_size, d, dropout))

        self.network = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C_in, T).

        Returns
        -------
        torch.Tensor
            Shape (B, C_out, T) where C_out = channels[-1].
        """
        return self.network(x)
