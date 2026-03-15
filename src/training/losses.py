"""BCM multi-task loss with uniform weights."""

from typing import Dict

import torch
import torch.nn as nn


class BCMMultiLoss(nn.Module):
    """Uniform-weighted MSE loss across PET, PCK, AET, CWD.

    Loss = MSE(PET) + MSE(PCK) + MSE(AET) + MSE(CWD)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss()

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights (uniform)."""
        return {"pet": 1.0, "pck": 1.0, "aet": 1.0, "cwd": 1.0}

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted multi-task loss.

        Parameters
        ----------
        predictions : dict
            Model outputs with keys 'pet', 'pck', 'aet', 'cwd'.
        targets : dict
            Ground truth with same keys.
        epoch : int
            Current epoch (0-indexed) for weight scheduling.

        Returns
        -------
        dict
            'total': total loss, plus per-variable losses.
        """
        weights = self.get_weights(epoch)
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for var in ["pet", "pck", "aet", "cwd"]:
            loss = self.mse(predictions[var], targets[var])
            losses[var] = loss
            total = total + weights[var] * loss

        losses["total"] = total
        losses["weights"] = weights
        return losses
