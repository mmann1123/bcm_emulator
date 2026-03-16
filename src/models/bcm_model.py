"""BCMEmulator: Three-stage hierarchical TCN model."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbone import TCNBackbone
from .heads import AETHead, PointwiseHead


class BCMEmulator(nn.Module):
    """Three-stage BCM emulator.

    Stage 1: Shared TCN backbone extracts temporal features.
    Stage 2: PET head (unconstrained) and PCK head (softplus >= 0).
    Stage 3: AET predicted unconstrained (AET <= PET enforced post-denorm).
             CWD = PET - AET (algebraic, no parameters).

    Input channels (15 continuous + 8 FVEG embed = 23 total):
        Dynamic (10): ppt, tmin, tmax, wet_days, ppt_intensity, srad, snow_frac, pck_prev, aet_prev, vpd
        Static (5): elevation, topo_solar, lat, lon, awc (tiled across T)
        FVEG embedding (8): from vegetation class ID lookup

    Parameters
    ----------
    backbone_cfg : dict
        Keyword arguments for TCNBackbone.
    """

    # Channel indices for teacher-forced inputs
    PCK_PREV_IDX = 7
    AET_PREV_IDX = 8

    def __init__(
        self,
        backbone_cfg: Optional[dict] = None,
        num_fveg_classes: int = 0,
        fveg_embed_dim: int = 8,
    ):
        super().__init__()
        if backbone_cfg is None:
            backbone_cfg = {}

        self.num_fveg_classes = num_fveg_classes
        self.fveg_embed_dim = fveg_embed_dim if num_fveg_classes > 0 else 0

        # FVEG vegetation embedding
        if num_fveg_classes > 0:
            self.fveg_embedding = nn.Embedding(num_fveg_classes, fveg_embed_dim)

        self.backbone = TCNBackbone(**backbone_cfg)
        bb_out = self.backbone.out_channels

        # Stage 2 heads
        self.pet_head = PointwiseHead(bb_out, activation="softplus")
        self.pck_head = PointwiseHead(bb_out, activation="softplus")

        # Stage 3 head: takes backbone + PET + PCK
        self.aet_head = AETHead(bb_out + 2)

    def _prepend_fveg(self, x: torch.Tensor, fveg_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Concatenate FVEG embedding to input tensor if fveg is configured."""
        if self.num_fveg_classes > 0 and fveg_ids is not None:
            B, C, T = x.shape
            fveg_embed = self.fveg_embedding(fveg_ids)        # (B, embed_dim)
            fveg_tiled = fveg_embed.unsqueeze(-1).expand(-1, -1, T)  # (B, embed_dim, T)
            x = torch.cat([x, fveg_tiled], dim=1)             # (B, C+embed_dim, T)
        return x

    def forward(
        self,
        x: torch.Tensor,
        tf_ratio: float = 1.0,
        gt_pck_prev: Optional[torch.Tensor] = None,
        gt_aet_prev: Optional[torch.Tensor] = None,
        fveg_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional teacher forcing.

        When tf_ratio < 1.0, uses autoregressive single-pass loop over timesteps.
        When tf_ratio == 1.0, runs full sequence in parallel (no loop needed).

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 13, T). Input features (9 dynamic + 4 continuous static).
        tf_ratio : float
            Fraction of timesteps using ground-truth for pck_prev/aet_prev.
            1.0 = all GT (teacher forcing), 0.0 = all predicted.
        gt_pck_prev : torch.Tensor, optional
            Shape (B, 1, T). Ground-truth PCK(t-1) for teacher forcing.
        gt_aet_prev : torch.Tensor, optional
            Shape (B, 1, T). Ground-truth AET(t-1) for teacher forcing.
        fveg_ids : torch.Tensor, optional
            Shape (B,). Integer FVEG class IDs for vegetation embedding.

        Returns
        -------
        dict
            Keys: 'pet', 'pck', 'aet', 'cwd', each (B, 1, T).
        """
        B, C, T = x.shape

        if tf_ratio >= 1.0 or gt_pck_prev is None or gt_aet_prev is None:
            # Full parallel forward -- no autoregressive loop needed
            return self._forward_parallel(x, fveg_ids)
        else:
            # Autoregressive with scheduled sampling
            return self._forward_autoregressive(x, tf_ratio, gt_pck_prev, gt_aet_prev, fveg_ids)

    def _forward_parallel(self, x: torch.Tensor, fveg_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Parallel forward pass (no teacher forcing substitution)."""
        x = self._prepend_fveg(x, fveg_ids)
        features = self.backbone(x)

        pet = self.pet_head(features)
        pck = self.pck_head(features)

        # AET head gets backbone features + PET + PCK
        aet_input = torch.cat([features, pet, pck], dim=1)
        aet = self.aet_head(aet_input, pet)

        cwd = pet - aet

        return {"pet": pet, "pck": pck, "aet": aet, "cwd": cwd}

    def _forward_autoregressive(
        self,
        x: torch.Tensor,
        tf_ratio: float,
        gt_pck_prev: torch.Tensor,
        gt_aet_prev: torch.Tensor,
        fveg_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive forward with scheduled sampling.

        Loops over timesteps, filling pck_prev and aet_prev channels
        from GT or model predictions based on Bernoulli(tf_ratio).
        """
        B, C, T = x.shape
        device = x.device

        # Output buffers
        pet_out = torch.zeros(B, 1, T, device=device)
        pck_out = torch.zeros(B, 1, T, device=device)
        aet_out = torch.zeros(B, 1, T, device=device)

        # Working copy of input -- we'll modify pck_prev and aet_prev channels
        x_buf = x.clone()

        for t in range(T):
            if t > 0:
                # Decide whether to use GT or predicted for this timestep
                use_gt = torch.rand(1, device=device).item() < tf_ratio

                if use_gt:
                    x_buf[:, self.PCK_PREV_IDX, t] = gt_pck_prev[:, 0, t]
                    x_buf[:, self.AET_PREV_IDX, t] = gt_aet_prev[:, 0, t]
                else:
                    x_buf[:, self.PCK_PREV_IDX, t] = pck_out[:, 0, t - 1]
                    x_buf[:, self.AET_PREV_IDX, t] = aet_out[:, 0, t - 1]

            # Run backbone on sequence up to t+1 (causal, so only sees <=t)
            # For efficiency, run on full sequence but only take output at t
            x_slice = self._prepend_fveg(x_buf[:, :, :t + 1], fveg_ids)
            features = self.backbone(x_slice)

            # Take features at timestep t
            feat_t = features[:, :, -1:]  # (B, C_bb, 1)

            pet_t = self.pet_head(feat_t)
            pck_t = self.pck_head(feat_t)

            aet_input_t = torch.cat([feat_t, pet_t, pck_t], dim=1)
            aet_t = self.aet_head(aet_input_t, pet_t)

            pet_out[:, :, t] = pet_t[:, :, 0]
            pck_out[:, :, t] = pck_t[:, :, 0]
            aet_out[:, :, t] = aet_t[:, :, 0]

        cwd_out = pet_out - aet_out

        return {"pet": pet_out, "pck": pck_out, "aet": aet_out, "cwd": cwd_out}
