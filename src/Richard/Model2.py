
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
)
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, Subset


# ============================================================
# U-Net building blocks
# ============================================================

class ConvBlock(nn.Module):
    """Two convolutions with normalisation and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """
    Configurable 2D U-Net.

    Input:  (batch, context_size, rows, cols)
    Output: (batch, 1, rows, cols)

    n_blocks controls the number of encoder and decoder blocks.
    """

    def __init__(
        self,
        in_channels=5,
        base_channels=32,
        n_blocks=4,
    ):
        super().__init__()

        if n_blocks < 1:
            raise ValueError("n_blocks must be at least 1.")

        self.n_blocks = n_blocks

        # Channel counts at each resolution.
        channels = [
            base_channels * (2 ** i)
            for i in range(n_blocks)
        ]

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Encoder
        self.encoder = nn.ModuleList()
        current_channels = in_channels

        for out_channels in channels:
            self.encoder.append(
                ConvBlock(current_channels, out_channels)
            )
            current_channels = out_channels

        # Bottleneck: one level deeper than the last encoder block.
        bottleneck_channels = channels[-1] * 2

        self.bottleneck = ConvBlock(
            channels[-1],
            bottleneck_channels,
        )

        # Decoder, traversing encoder levels in reverse.
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        current_channels = bottleneck_channels

        for skip_channels in reversed(channels):
            self.upconvs.append(
                nn.Conv2d(
                    current_channels,
                    skip_channels,
                    kernel_size=1,
                )
            )

            self.decoder.append(
                ConvBlock(
                    skip_channels + skip_channels,
                    skip_channels,
                )
            )

            current_channels = skip_channels

        self.out_conv = nn.Conv2d(
            base_channels,
            1,
            kernel_size=1,
        )

    def forward(self, x):
        # Encoder and save skip connections.
        skips = []

        for i, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)

            if i < self.n_blocks - 1:
                x = self.pool(x)

        # Downsample once more before the bottleneck.
        x = self.pool(x)
        x = self.bottleneck(x)

        # Decoder.
        for upconv, block, skip in zip(
            self.upconvs,
            self.decoder,
            reversed(skips),
        ):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            x = upconv(x)
            x = block(torch.cat([x, skip], dim=1))

        return F.softplus(self.out_conv(x))


class SPECTUNet(pl.LightningModule):
    """Lightning wrapper for configurable U-Net denoising."""

    def __init__(
        self,
        context_size=5,
        base_channels=32,
        n_blocks=4,
        n_res_blocks=3,
        dropout=0.0,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.context_size = context_size
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        self.lr = lr

        self.net = UNet(
            in_channels=context_size,
            base_channels=base_channels,
            n_blocks=n_blocks,
        )

        #No res blocks or dropout in this model
        #Still in model definition to make code the same

    def forward(self, x):
        x = x.clamp_min(0)
        x = torch.log1p(x)

        log_prediction = self.net(x)

        return torch.expm1(log_prediction)

    @staticmethod
    def loss_fn(pred, target):
        pred_log = torch.log1p(pred.clamp_min(0))
        target_log = torch.log1p(target.clamp_min(0))

        return F.smooth_l1_loss(pred_log, target_log)

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            return batch["mixed"], batch["truth"]

        return batch

    def training_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)

        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)

        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)

        pred = self(x).clamp_min(0)
        target = y.clamp_min(0)

        # Flatten over batch, channel and spatial dimensions.
        error = pred - target

        # Basic pixel-wise metrics.
        mae = error.abs().mean()
        rmse = torch.sqrt(error.square().mean())
        bias = error.mean()

        # Relative error, avoiding division by zero.
        epsilon = 1e-8
        relative_mae = (
            error.abs() / target.clamp_min(epsilon)
        ).mean()

        # Total-count recovery, per image.
        pred_counts = pred.sum(dim=(1, 2, 3))
        true_counts = target.sum(dim=(1, 2, 3))

        count_recovery = (
            pred_counts / true_counts.clamp_min(epsilon)
        ).mean()

        count_bias = (
            (pred_counts - true_counts)
            / true_counts.clamp_min(epsilon)
        ).mean()

        # PSNR: use the target's dynamic range.
        # For count-valued images, this is more interpretable than
        # assuming a fixed maximum count.
        data_range = (
            target.amax(dim=(1, 2, 3), keepdim=True)
            - target.amin(dim=(1, 2, 3), keepdim=True)
        ).clamp_min(epsilon)

        mse_per_image = error.square().mean(
            dim=(1, 2, 3), keepdim=True
        )
        psnr = (
            10 * torch.log10(
                data_range.square() / mse_per_image.clamp_min(epsilon)
            )
        ).mean()

        # SSIM measures local structural similarity.
        # It is averaged across images.
        target_min = target.amin(dim=(1, 2, 3), keepdim=True)
        target_max = target.amax(dim=(1, 2, 3), keepdim=True)
        target_range = (target_max - target_min).clamp_min(epsilon)

        # SSIM expects the inputs to share an intensity scale.
        pred_normalised = (pred - target_min) / target_range
        target_normalised = (target - target_min) / target_range

        ssim = structural_similarity_index_measure(
            pred_normalised,
            target_normalised,
            data_range=1.0,
        )

        # Original training loss, calculated in log-count space.
        test_loss = self.loss_fn(pred, target)

        metrics = {
            "test_loss": test_loss,
            "test_MAE": mae,
            "test_RMSE": rmse,
            "test_bias": bias,
            "test_relative_MAE": relative_mae,
            "test_PSNR_dB": psnr,
            "test_SSIM": ssim,
            "test_count_recovery": count_recovery,
            "test_relative_count_bias": count_bias,
        }

        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name in ["test_loss", "test_MAE", "test_SSIM"]),
                batch_size=x.shape[0],
            )

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

    def export_to_state_dict(self, path):
        torch.save(self.state_dict(), path)
        print(f"Saved state_dict to {path}")

    @classmethod
    def load_from_state_dict(cls, path, **kwargs):
        model = cls(**kwargs)

        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state_dict)
        model.eval()

        return model

# ============================================================
# Loss tracker
# ============================================================

class LossTracker(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get("train_loss")

        if value is not None:
            self.train_losses.append(value.detach().cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get("val_loss")

        if value is not None:
            self.val_losses.append(value.detach().cpu().item())
