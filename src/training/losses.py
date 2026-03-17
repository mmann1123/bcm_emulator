"""BCM multi-task loss with configurable scheduled weights."""

from typing import Dict, List

import torch
import torch.nn as nn


class BCMMultiLoss(nn.Module):
    """Weighted Huber loss across PET, PCK, AET, CWD with optional PET decay schedule
    and extreme-aware MSE penalty.

    Loss = Σ w_var*Huber(var) + extreme_weight * MSE_extreme(extreme_vars)
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
        delta: float = 1.35,
        extreme_threshold: float = 1.28,
        extreme_weight: float = 0.0,
        extreme_vars: List[str] = None,
        extreme_asym: float = 1.5,
        **kwargs,
    ):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.pet_initial = pet_initial
        self.pck_initial = pck_initial
        self.aet_initial = aet_initial
        self.cwd_initial = cwd_initial
        self.pet_decay = pet_decay
        self.pet_floor = pet_floor
        self.total_epochs = total_epochs
        self.extreme_threshold = extreme_threshold
        self.extreme_weight = extreme_weight
        self.extreme_vars = extreme_vars or []
        self.extreme_asym = extreme_asym

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
            loss = self.huber(predictions[var], targets[var])
            losses[var] = loss
            total = total + weights[var] * loss

        # Extreme-aware penalty (additive MSE on tail samples)
        if self.extreme_weight > 0:
            for var in self.extreme_vars:
                if var not in predictions:
                    continue
                pred, tgt = predictions[var], targets[var]
                extreme_mask = (tgt > self.extreme_threshold).float()
                n_extreme = extreme_mask.sum().clamp(min=1.0)
                sq_err = (pred - tgt) ** 2
                # Asymmetric: penalize underprediction (pred < tgt) more
                asym = torch.where(
                    pred < tgt,
                    torch.tensor(self.extreme_asym, device=pred.device),
                    torch.tensor(1.0 / self.extreme_asym, device=pred.device),
                )
                extreme_loss = (sq_err * asym * extreme_mask).sum() / n_extreme
                losses[f"{var}_extreme"] = extreme_loss
                total = total + self.extreme_weight * extreme_loss

        losses["total"] = total
        losses["weights"] = weights
        return losses
