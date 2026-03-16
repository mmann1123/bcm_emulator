"""Training loop for BCM emulator."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from contextlib import nullcontext

try:
    # PyTorch >= 2.0
    from torch.amp import GradScaler, autocast
    _HAS_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _HAS_NEW_AMP = False
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ..models.bcm_model import BCMEmulator
from .losses import BCMMultiLoss
from .teacher_forcing import get_tf_ratio

logger = logging.getLogger(__name__)


class BCMTrainer:
    """Training loop for BCM emulator with curriculum scheduling.

    Parameters
    ----------
    model : BCMEmulator
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    cfg : ConfigNamespace
        Training configuration.
    device : torch.device
        Device to train on.
    """

    def __init__(
        self,
        model: BCMEmulator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        tcfg = cfg.training
        self.epochs = tcfg.epochs
        self.amp_enabled = tcfg.amp
        self.grad_clip = tcfg.grad_clip_norm

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=tcfg.learning_rate,
            weight_decay=tcfg.weight_decay,
        )

        # Scheduler: linear warmup + cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=tcfg.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - tcfg.warmup_epochs
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[tcfg.warmup_epochs],
        )

        # Loss
        lw = tcfg.loss_weights
        self.criterion = BCMMultiLoss(
            pet_initial=lw.pet_initial,
            pck_initial=lw.pck_initial,
            aet_initial=lw.aet_initial,
            cwd_initial=lw.cwd_initial,
            pet_decay=lw.pet_decay,
            pet_floor=lw.pet_floor,
            total_epochs=self.epochs,
        )

        # AMP — disable on CPU
        if self.device.type != "cuda":
            self.amp_enabled = False
        self.scaler = GradScaler(enabled=self.amp_enabled)

        # Teacher forcing
        self.tf_warmup_fraction = tcfg.teacher_forcing.warmup_fraction

        # Checkpoint
        self.checkpoint_dir = Path(cfg.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")

    def train(self) -> Dict:
        """Run full training loop."""
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            tf_ratio = get_tf_ratio(epoch, self.epochs, self.tf_warmup_fraction)
            weights = self.criterion.get_weights(epoch)

            # Train
            train_losses = self._train_epoch(epoch, tf_ratio)
            val_losses = self._validate(epoch, tf_ratio)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"TF={tf_ratio:.2f} | LR={lr:.6f} | "
                f"Train: {train_losses['total']:.4f} "
                f"(PET={train_losses['pet']:.4f} PCK={train_losses['pck']:.4f} "
                f"AET={train_losses['aet']:.4f} CWD={train_losses['cwd']:.4f}) | "
                f"Val: {val_losses['total']:.4f} | "
                f"Weights: PET={weights['pet']:.2f} PCK={weights['pck']:.2f}"
            )

            history["train_loss"].append(train_losses["total"])
            history["val_loss"].append(val_losses["total"])

            # Checkpoint
            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self._save_checkpoint(epoch, val_losses["total"], is_best=True)

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_losses["total"], is_best=False)

        return history

    def _train_epoch(self, epoch: int, tf_ratio: float) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running = {"total": 0.0, "pet": 0.0, "pck": 0.0, "aet": 0.0, "cwd": 0.0}
        n_batches = 0

        for batch in self.train_loader:
            inputs = batch["inputs"].to(self.device)         # (B, 15, T)
            targets = {k: batch["targets"][k].to(self.device) for k in batch["targets"]}
            gt_pck_prev = batch.get("gt_pck_prev")
            gt_aet_prev = batch.get("gt_aet_prev")
            fveg_ids = batch.get("fveg_ids")
            if gt_pck_prev is not None:
                gt_pck_prev = gt_pck_prev.to(self.device)
            if gt_aet_prev is not None:
                gt_aet_prev = gt_aet_prev.to(self.device)
            if fveg_ids is not None:
                fveg_ids = fveg_ids.to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.amp_enabled) if (_HAS_NEW_AMP and self.amp_enabled) else nullcontext():
                preds = self.model(inputs, tf_ratio, gt_pck_prev, gt_aet_prev, fveg_ids)
                losses = self.criterion(preds, targets, epoch)

            self.scaler.scale(losses["total"]).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def _validate(self, epoch: int, tf_ratio: float) -> Dict[str, float]:
        """Validate for one epoch (always fully autoregressive)."""
        self.model.eval()
        running = {"total": 0.0, "pet": 0.0, "pck": 0.0, "aet": 0.0, "cwd": 0.0}
        n_batches = 0

        for batch in self.val_loader:
            inputs = batch["inputs"].to(self.device)
            targets = {k: batch["targets"][k].to(self.device) for k in batch["targets"]}
            gt_pck_prev = batch.get("gt_pck_prev")
            gt_aet_prev = batch.get("gt_aet_prev")
            fveg_ids = batch.get("fveg_ids")
            if gt_pck_prev is not None:
                gt_pck_prev = gt_pck_prev.to(self.device)
            if gt_aet_prev is not None:
                gt_aet_prev = gt_aet_prev.to(self.device)
            if fveg_ids is not None:
                fveg_ids = fveg_ids.to(self.device)

            with autocast("cuda", enabled=self.amp_enabled) if (_HAS_NEW_AMP and self.amp_enabled) else nullcontext():
                # Validation always uses fully autoregressive mode
                preds = self.model(inputs, 0.0, gt_pck_prev, gt_aet_prev, fveg_ids)
                losses = self.criterion(preds, targets, epoch)

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")
