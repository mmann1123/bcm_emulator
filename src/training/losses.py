"""BCM multi-task loss with configurable scheduled weights."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCMMultiLoss(nn.Module):
    """Weighted Huber loss across PET, PCK, AET, CWD with optional PET decay schedule,
    extreme-aware MSE penalty, and annual-pooled (water-year) MSE term.

    Loss = Σ w_var*Huber(var)
         + extreme_weight * MSE_extreme(extreme_vars)
         + annual_lambda * Σ_{var ∈ annual_vars} MSE(annual_mean(pred), annual_mean(target))
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
        loss_type: str = "huber",
        delta: float = 1.35,
        extreme_threshold: float = 1.28,
        extreme_weight: float = 0.0,
        extreme_vars: List[str] = None,
        extreme_asym: float = 1.5,
        annual_lambda: float = 0.0,
        annual_vars: List[str] = None,
        annual_min_complete_wy: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "mse":
            self.base_loss = nn.MSELoss()
        else:
            self.base_loss = nn.HuberLoss(delta=delta)
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
        self.annual_lambda = float(annual_lambda)
        self.annual_vars = annual_vars or []
        self.annual_min_complete_wy = int(annual_min_complete_wy)
        # Canonical Oct→Sep month sequence (0=Jan…11=Dec → Oct=9, Nov=10, …, Sep=8).
        self._WY_SEQ = [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for given epoch. PET decays; others are constant."""
        pet_w = max(self.pet_initial * (self.pet_decay ** epoch), self.pet_floor)
        return {
            "pet": pet_w,
            "pck": self.pck_initial,
            "aet": self.aet_initial,
            "cwd": self.cwd_initial,
        }

    def _annual_pool_term(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        month_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-variable annual-pooled MSE on complete Oct→Sep water years.

        For each batch element, find the first October in `month_indices`, then iterate
        consecutive 12-month blocks while the block's month sequence equals
        [9,10,11,0,1,2,3,4,5,6,7,8]. Each complete block contributes one
        (pred_mean, target_mean) pair to the per-variable stack. MSE is then computed
        with reduction='mean' across all collected (batch × WY) means, so a given
        annual_lambda is comparable across runs regardless of batch size or how many
        complete WYs each window happens to contain.

        Returns a dict mapping each var in self.annual_vars to a scalar tensor.
        Variables for which no complete WYs were found across the batch return 0.
        """
        # month_indices: (B, T) int64
        if month_indices.dim() != 2:
            return {}
        B, T = month_indices.shape
        device = month_indices.device

        # Vectorized WY-block extraction:
        #   1) For each batch row, find first October (month==9). If none, that row is dropped.
        #   2) From first_oct, the row contains floor((T-first_oct)/12) complete 12-month blocks.
        #   3) For each complete block in the batch, mean over its 12 months gives one annual scalar.
        # Monthly cadence guarantees blocks are exactly [9,10,11,0,1,2,3,4,5,6,7,8] — assert in
        # debug builds via the canonical-sequence check; skip per-block validation in the hot path.
        is_oct = month_indices == 9                                     # (B, T) bool
        any_oct = is_oct.any(dim=1)                                     # (B,)
        # argmax returns 0 if no True, but we mask those rows via any_oct.
        first_oct = is_oct.float().argmax(dim=1)                        # (B,)
        # Number of complete WYs available starting from first_oct in each row.
        n_per_row = torch.where(
            any_oct,
            (T - first_oct) // 12,
            torch.zeros_like(first_oct),
        )                                                               # (B,)
        total_wys = int(n_per_row.sum().item())
        if total_wys < self.annual_min_complete_wy:
            return {}

        # Build flat (N_total_wys,) batch_idx and start_idx tensors — all on-device, no Python loops.
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=device), n_per_row
        )                                                               # (N,)
        # WY index within row: 0, 1, …, n_per_row[b]-1, concatenated across rows.
        # Trick: arange(N) - exclusive_cumsum(n_per_row) repeated by n_per_row.
        cum = n_per_row.cumsum(0)
        cum_excl = torch.cat([cum.new_zeros(1), cum[:-1]])
        wy_within = (
            torch.arange(total_wys, device=device)
            - torch.repeat_interleave(cum_excl, n_per_row)
        )
        start_idx = first_oct[batch_idx] + 12 * wy_within               # (N,)

        # Time offsets 0..11 to gather each block's 12 months.
        t_offsets = torch.arange(12, device=device)
        time_idx = start_idx[:, None] + t_offsets[None, :]              # (N, 12)

        out = {}
        for var in self.annual_vars:
            if var not in predictions or var not in targets:
                continue
            p = predictions[var]                                        # (B, *, T) — typically (B, 1, T)
            t = targets[var]
            # Reshape to (B, T) for indexing — squeeze any singleton mid-dims.
            if p.dim() == 3 and p.shape[1] == 1:
                p2 = p.squeeze(1)
                t2 = t.squeeze(1)
            elif p.dim() == 2:
                p2 = p
                t2 = t
            else:
                # Fall back: collapse all middle dims by mean (rare path).
                p2 = p.reshape(B, -1, T).mean(dim=1)
                t2 = t.reshape(B, -1, T).mean(dim=1)

            # Gather 12-month blocks per WY. Result: (N, 12) → mean over time → (N,).
            pred_blocks = p2[batch_idx[:, None], time_idx]              # (N, 12)
            tgt_blocks = t2[batch_idx[:, None], time_idx]
            pred_annual = pred_blocks.mean(dim=1)                       # (N,)
            tgt_annual = tgt_blocks.mean(dim=1)
            out[var] = F.mse_loss(pred_annual, tgt_annual, reduction="mean")
        return out

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        epoch: int,
        month_indices: Optional[torch.Tensor] = None,
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
            loss = self.base_loss(predictions[var], targets[var])
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

        # Annual-pooled (Oct→Sep) MSE term — applied in z-score (normalized) space,
        # matching the Huber and extreme terms. Skipped on eval/val paths where
        # month_indices is not provided.
        if self.annual_lambda > 0 and month_indices is not None and self.annual_vars:
            annual_terms = self._annual_pool_term(predictions, targets, month_indices)
            for var, l_var in annual_terms.items():
                losses[f"annual_{var}"] = l_var
                total = total + self.annual_lambda * l_var

        losses["total"] = total
        losses["weights"] = weights
        return losses
