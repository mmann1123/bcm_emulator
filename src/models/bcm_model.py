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

    KBDI is excluded from the backbone and injected directly into the AET head
    to act as a drought-stress inhibitor without polluting PET/PCK encoding.

    Input channels (22 continuous + 8 FVEG embed = 30 backbone → in_channels=35 with SWS/rollstd):
        Dynamic (14, excl KBDI): ppt, tmin, tmax, wet_days, ppt_intensity, srad, snow_frac, pck_prev, aet_prev, vpd, sws, vpd_roll6_std, srad_roll6_std, tmax_roll3_std
        Static (13): elev, topo_solar, lat, lon, ksat, sand, clay, soil_depth, aridity, FC, WP, SOM, windward
        FVEG embedding (8): from vegetation class ID lookup
    KBDI (1): passed separately to AET head only

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
        aet_backbone_cfg: Optional[dict] = None,
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

        # AET sub-backbone (optional dual-backbone mode)
        self.has_aet_backbone = aet_backbone_cfg is not None
        if self.has_aet_backbone:
            n_dyn = 14  # dynamic channels in backbone input (excl KBDI)
            # Extract and remove routing keys before passing to TCNBackbone
            aet_backbone_cfg = dict(aet_backbone_cfg)  # don't mutate caller's dict
            self.aet_dyn_idx = list(aet_backbone_cfg.pop("dyn_channels"))
            static_channels = list(aet_backbone_cfg.pop("static_channels"))
            self.aet_static_idx = [n_dyn + s for s in static_channels]
            # Channel indices into raw x (before FVEG prepend) — dynamic + static only
            self.aet_raw_channel_idx = self.aet_dyn_idx + self.aet_static_idx

            # Separate FVEG embedding for AET sub-backbone (gradient isolation)
            if num_fveg_classes > 0:
                self.aet_fveg_embedding = nn.Embedding(num_fveg_classes, fveg_embed_dim)
                # Initialize with same weights as main embedding
                self.aet_fveg_embedding.weight.data.copy_(self.fveg_embedding.weight.data)
                n_aet_in = len(self.aet_raw_channel_idx) + fveg_embed_dim
            else:
                n_aet_in = len(self.aet_raw_channel_idx)

            aet_backbone_cfg["in_channels"] = n_aet_in
            self.aet_backbone = TCNBackbone(**aet_backbone_cfg)
            aet_bb_out = bb_out + self.aet_backbone.out_channels
        else:
            aet_bb_out = bb_out

        # Stage 2 heads (main backbone only)
        self.pet_head = PointwiseHead(bb_out, activation="softplus")
        self.pck_head = PointwiseHead(bb_out, activation="softplus")

        # Stage 3 head: stress-fraction architecture (stress × Kv × PET + correction)
        self.aet_head = AETHead(aet_bb_out)

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
        kbdi: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional teacher forcing.

        When tf_ratio < 1.0, uses autoregressive single-pass loop over timesteps.
        When tf_ratio == 1.0, runs full sequence in parallel (no loop needed).

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, T). Input features (14 dynamic + 13 static, no KBDI).
        tf_ratio : float
            Fraction of timesteps using ground-truth for pck_prev/aet_prev.
            1.0 = all GT (teacher forcing), 0.0 = all predicted.
        gt_pck_prev : torch.Tensor, optional
            Shape (B, 1, T). Ground-truth PCK(t-1) for teacher forcing.
        gt_aet_prev : torch.Tensor, optional
            Shape (B, 1, T). Ground-truth AET(t-1) for teacher forcing.
        fveg_ids : torch.Tensor, optional
            Shape (B,). Integer FVEG class IDs for vegetation embedding.
        kbdi : torch.Tensor, optional
            Shape (B, 1, T). KBDI routed directly to AET head.
        kv : torch.Tensor, optional
            Shape (B, 1, T). Monthly Kv crop coefficient routed to AET head.

        Returns
        -------
        dict
            Keys: 'pet', 'pck', 'aet', 'cwd', each (B, 1, T).
        """
        B, C, T = x.shape

        if tf_ratio >= 1.0 or gt_pck_prev is None or gt_aet_prev is None:
            return self._forward_parallel(x, fveg_ids, kbdi, kv)
        else:
            return self._forward_autoregressive(x, tf_ratio, gt_pck_prev, gt_aet_prev, fveg_ids, kbdi, kv)

    def _build_aet_input(self, x_raw: torch.Tensor, fveg_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Build AET sub-backbone input from raw x (before main FVEG prepend).

        Extracts selected dynamic+static channels and appends the AET-specific
        FVEG embedding (separate from the main backbone's embedding).
        """
        aet_input = x_raw[:, self.aet_raw_channel_idx, :]
        if self.num_fveg_classes > 0 and fveg_ids is not None:
            T = x_raw.shape[-1]
            aet_fveg = self.aet_fveg_embedding(fveg_ids)  # (B, embed_dim)
            aet_fveg_tiled = aet_fveg.unsqueeze(-1).expand(-1, -1, T)
            aet_input = torch.cat([aet_input, aet_fveg_tiled], dim=1)
        return aet_input

    def _forward_parallel(self, x: torch.Tensor, fveg_ids: Optional[torch.Tensor] = None, kbdi: Optional[torch.Tensor] = None, kv: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Parallel forward pass (no teacher forcing substitution)."""
        # AET sub-backbone (from raw x, before main FVEG prepend)
        if self.has_aet_backbone:
            aet_input = self._build_aet_input(x, fveg_ids)

        x = self._prepend_fveg(x, fveg_ids)
        features = self.backbone(x)

        if self.has_aet_backbone:
            aet_features = self.aet_backbone(aet_input)
            aet_combined = torch.cat([features, aet_features], dim=1)
        else:
            aet_combined = features

        pet = self.pet_head(features)
        pck = self.pck_head(features)

        # AET stress-fraction head
        _kbdi = kbdi if kbdi is not None else torch.zeros_like(pet)
        _kv = kv if kv is not None else torch.zeros_like(pet)
        aet = self.aet_head(aet_combined, pet, pck, _kbdi, _kv)

        cwd = pet - aet

        return {"pet": pet, "pck": pck, "aet": aet, "cwd": cwd}

    def _forward_autoregressive(
        self,
        x: torch.Tensor,
        tf_ratio: float,
        gt_pck_prev: torch.Tensor,
        gt_aet_prev: torch.Tensor,
        fveg_ids: Optional[torch.Tensor] = None,
        kbdi: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
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
            x_raw_slice = x_buf[:, :, :t + 1]
            x_slice = self._prepend_fveg(x_raw_slice, fveg_ids)
            features = self.backbone(x_slice)

            # Take features at timestep t
            feat_t = features[:, :, -1:]  # (B, C_bb, 1)

            # AET sub-backbone (from raw slice, separate FVEG embedding)
            if self.has_aet_backbone:
                aet_slice = self._build_aet_input(x_raw_slice, fveg_ids)
                aet_features = self.aet_backbone(aet_slice)
                aet_feat_t = aet_features[:, :, -1:]
                aet_combined_t = torch.cat([feat_t, aet_feat_t], dim=1)
            else:
                aet_combined_t = feat_t

            pet_t = self.pet_head(feat_t)
            pck_t = self.pck_head(feat_t)

            # AET stress-fraction head
            kbdi_t = kbdi[:, :, t:t + 1] if kbdi is not None else torch.zeros_like(pet_t)
            kv_t = kv[:, :, t:t + 1] if kv is not None else torch.zeros_like(pet_t)
            aet_t = self.aet_head(aet_combined_t, pet_t, pck_t, kbdi_t, kv_t)

            pet_out[:, :, t] = pet_t[:, :, 0]
            pck_out[:, :, t] = pck_t[:, :, 0]
            aet_out[:, :, t] = aet_t[:, :, 0]

        cwd_out = pet_out - aet_out

        return {"pet": pet_out, "pck": pck_out, "aet": aet_out, "cwd": cwd_out}
