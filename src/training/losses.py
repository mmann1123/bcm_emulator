"""BCM multi-task loss with configurable scheduled weights."""

from typing import Dict

import torch
import torch.nn as nn


class BCMMultiLoss(nn.Module):
    """Weighted MSE loss across PET, PCK, AET, CWD with optional PET decay schedule.

    Loss = w_pet*MSE(PET) + w_pck*MSE(PCK) + w_aet*MSE(AET) + w_cwd*MSE(CWD)
    """

    def __init__(
        self,
        pet_initial: float = 1.0,
        pck_initial: float = 1.0,
        aet_initial: float = 2.0,
        cwd_initial: float = 2.0,
        pet_decay: float = 1.0,
        pet_floor: float = 0.5,
        total_epochs: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pet_initial = pet_initial
        self.pck_initial = pck_initial
        self.aet_initial = aet_initial
        self.cwd_initial = cwd_initial
        self.pet_decay = pet_decay
        self.pet_floor = pet_floor
        self.total_epochs = total_epochs

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for given epoch. PET decays; others are constant."""
        pet_w = max(self.pet_initial * (self.pet_decay ** epoch), self.pet_floor)
        return {
            "pet": pet_w,
            "pck": self.pck_initial,
            "aet": self.aet_initial,
            "cwd": self.cwd_initial,
        }

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
