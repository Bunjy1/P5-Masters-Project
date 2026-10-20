
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
# Residual block
# ============================================================

class ResNetBlock2D(nn.Module):
    """Two convolutions with a residual connection."""

    def __init__(self, channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)


# ============================================================
# Encoder block
# ============================================================

class UNetDown2D(nn.Module):
    """Strided convolution for downsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        normalize=True,
        dropout=0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# ============================================================
# Decoder block
# ============================================================

class UNetUp2D(nn.Module):
    """Transposed convolution followed by skip concatenation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        normalize=True,
        dropout=0.0,
    ):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.decoder(x)

        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Upsampled shape {x.shape[-2:]} does not match "
                f"skip shape {skip.shape[-2:]}. "
                "Use input dimensions divisible by "
                f"{2 ** (self.n_blocks + 1)}."
            )

        return torch.cat([x, skip], dim=1)


# ============================================================
# Configurable residual U-Net
# ============================================================

class UNet(nn.Module):
    """
    Residual-bottleneck U-Net.

    Input:  (batch, in_channels, height, width)
    Output: (batch, 1, height, width)

    n_blocks:
        Number of encoder downsampling stages and skip levels.

    n_res_blocks:
        Number of residual blocks at the bottleneck.

    base_channels:
        Number of channels in the first encoder stage.

    dropout:
        Dropout probability in deeper encoder/decoder stages
        and the bottleneck downsampling block.
    """

    def __init__(
        self,
        in_channels=5,
        base_channels=32,
        n_blocks=4,
        n_res_blocks=6,
        dropout=0.0,
    ):
        super().__init__()

        if n_blocks < 1:
            raise ValueError("n_blocks must be at least 1.")

        if n_res_blocks < 0:
            raise ValueError("n_res_blocks must be non-negative.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.n_blocks = n_blocks

        channels = [
            base_channels * (2 ** i)
            for i in range(n_blocks)
        ]

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.ModuleList()
        current_channels = in_channels

        for i, out_channels in enumerate(channels):
            self.encoder.append(
                UNetDown2D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    normalize=(i != 0),
                    dropout=dropout if i >= 3 else 0.0,
                )
            )
            current_channels = out_channels

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        bottleneck_channels = channels[-1] * 2

        self.bottleneck_down = UNetDown2D(
            in_channels=channels[-1],
            out_channels=bottleneck_channels,
            dropout=dropout,
        )

        self.resnet = nn.Sequential(
            *[
                ResNetBlock2D(bottleneck_channels)
                for _ in range(n_res_blocks)
            ]
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.ModuleList()
        current_channels = bottleneck_channels

        for i, skip_channels in enumerate(reversed(channels)):
            self.decoder.append(
                UNetUp2D(
                    in_channels=current_channels,
                    out_channels=skip_channels,
                    dropout=dropout if i < 2 else 0.0,
                )
            )

            # Decoder output is concatenated with its skip.
            current_channels = skip_channels * 2

        # Final upsampling restores the original resolution.
        self.final_up = nn.ConvTranspose2d(
            in_channels=base_channels * 2,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Original input is concatenated with the final output.
        self.final_conv = nn.Conv2d(
            in_channels=1 + in_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        original_input = x

        # Encoder.
        skips = []

        for block in self.encoder:
            x = block(x)
            skips.append(x)

        # Bottleneck.
        x = self.bottleneck_down(x)
        x = self.resnet(x)

        # Decoder.
        for block, skip in zip(
            self.decoder,
            reversed(skips),
        ):
            x = block(x, skip)

        # Return to full resolution.
        x = self.final_up(x)

        if x.shape[-2:] != original_input.shape[-2:]:
            raise ValueError(
                "Output resolution does not match input resolution. "
                f"Input: {original_input.shape[-2:]}, "
                f"output: {x.shape[-2:]}. "
                "Use height and width divisible by "
                f"{2 ** (self.n_blocks + 1)}."
            )

        # Final skip from the original input.
        x = torch.cat([x, original_input], dim=1)

        return F.softplus(self.final_conv(x))


# ============================================================
# Lightning wrapper
# ============================================================

class SPECTUNet(pl.LightningModule):
    """Lightning wrapper for residual U-Net denoising."""

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
        self.n_res_blocks = n_res_blocks
        self.dropout = dropout
        self.lr = lr

        self.net = UNet(
            in_channels=context_size,
            base_channels=base_channels,
            n_blocks=n_blocks,
            n_res_blocks=n_res_blocks,
            dropout=dropout,
        )

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

        error = pred - target

        mae = error.abs().mean()
        rmse = torch.sqrt(error.square().mean())
        bias = error.mean()

        epsilon = 1e-8

        relative_mae = (
            error.abs() / target.clamp_min(epsilon)
        ).mean()

        pred_counts = pred.sum(dim=(1, 2, 3))
        true_counts = target.sum(dim=(1, 2, 3))

        count_recovery = (
            pred_counts / true_counts.clamp_min(epsilon)
        ).mean()

        count_bias = (
            (pred_counts - true_counts)
            / true_counts.clamp_min(epsilon)
        ).mean()

        data_range = (
            target.amax(dim=(1, 2, 3), keepdim=True)
            - target.amin(dim=(1, 2, 3), keepdim=True)
        ).clamp_min(epsilon)

        mse_per_image = error.square().mean(
            dim=(1, 2, 3), keepdim=True
        )

        psnr = (
            10 * torch.log10(
                data_range.square()
                / mse_per_image.clamp_min(epsilon)
            )
        ).mean()

        target_min = target.amin(
            dim=(1, 2, 3), keepdim=True
        )
        target_max = target.amax(
            dim=(1, 2, 3), keepdim=True
        )
        target_range = (
            target_max - target_min
        ).clamp_min(epsilon)

        pred_normalised = (
            pred - target_min
        ) / target_range

        target_normalised = (
            target - target_min
        ) / target_range

        from torchmetrics.functional.image import (
            structural_similarity_index_measure,
        )

        ssim = structural_similarity_index_measure(
            pred_normalised,
            target_normalised,
            data_range=1.0,
        )

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
                prog_bar=name in [
                    "test_loss", "test_MAE", "test_SSIM"
                ],
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
