"""BCM multi-task loss with epoch-decaying weights."""

from typing import Dict

import torch
import torch.nn as nn


class BCMMultiLoss(nn.Module):
    """Weighted MSE loss across PET, PCK, AET, CWD with epoch-based weight scheduling.

    Loss = w_pet * MSE(PET) + w_pck * MSE(PCK) + w_aet * MSE(AET) + w_cwd * MSE(CWD)

    Weight schedule:
    - Epochs 1 to E/2 (teacher forcing): PET decays by 0.99/epoch to floor 0.5.
      PCK held at 2.0.
    - Epochs E/2 to E (scheduled sampling): PET=1.0, PCK=2.0, AET=1.0, CWD=0.5.
    """

    def __init__(
        self,
        pet_initial: float = 2.0,
        pck_initial: float = 2.0,
        aet_initial: float = 1.0,
        cwd_initial: float = 0.5,
        pet_decay: float = 0.99,
        pet_floor: float = 0.5,
        total_epochs: int = 100,
    ):
        super().__init__()
        self.pet_initial = pet_initial
        self.pck_initial = pck_initial
        self.aet_initial = aet_initial
        self.cwd_initial = cwd_initial
        self.pet_decay = pet_decay
        self.pet_floor = pet_floor
        self.total_epochs = total_epochs
        self.warmup_epochs = total_epochs // 2

        self.mse = nn.MSELoss()

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for a given epoch (0-indexed)."""
        if epoch < self.warmup_epochs:
            # Teacher forcing phase: decay PET, hold PCK
            pet_w = max(self.pet_floor, self.pet_initial * (self.pet_decay ** epoch))
            pck_w = self.pck_initial  # held constant
            aet_w = self.aet_initial
            cwd_w = self.cwd_initial
        else:
            # Scheduled sampling phase
            pet_w = 1.0
            pck_w = self.pck_initial  # held at 2.0
            aet_w = self.aet_initial
            cwd_w = self.cwd_initial

        return {"pet": pet_w, "pck": pck_w, "aet": aet_w, "cwd": cwd_w}

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
