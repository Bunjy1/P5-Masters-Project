import os
from pathlib import Path

import h5py
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from SPECTDataset import SPECTDataset

#Change between Model and Model2 to switch between model types
from Model import SPECTUNet, LossTracker

print('')


# ============================================================
# Configuration
# ============================================================

data_file = "/scratch/2758304h/data_repo/SPECTData/spect_dataset.h5"

# Set to false if you don't want to do the training
# e.g. to recompute metrics, make plots and so on
doTraining = True

# Dataset
context_size = 3       # Number of projection angles supplied as input
noise_fractions = (0.5, 0.25, 0.1) # Train only on noisy inputs, not fraction=1.0

# U-Net architecture
base_channels = 32      # Channels in the first encoder block
n_blocks = 2            # Number of encoder and decoder blocks
# Not used in Model2
n_res_blocks = 3        # Number of residual blocks in bottleneck
dropout = 0.0           # Dropout percentage to avoid overfitting

# Training
batch_size = 256        # Number of images loaded at a time
n_epochs = 100
lr = 1e-3

train_fraction = 0.8
val_fraction = 0.1
test_fraction = 0.1

seed = 12345
num_workers_cuda = 4

# Use to restrict amount of samples
# Useful to quickly test things out
max_train_samples = None           # e.g. 1000  
max_test_samples = None            # e.g.1000
max_val_samples = None             # e.g.1000

# Output directory for plots
output_dir = Path("spect_training")
output_dir.mkdir(parents=True, exist_ok=True)

# String to separate different model configurations
model_name = (
    f"spect_unet_context{context_size}"
    f"_blocks{n_blocks}_channels{base_channels}"
)

# path to save model and checkpoints during training
state_dict_path = output_dir / f"{model_name}_state_dict.pt"
checkpoint_dir = output_dir / "checkpoints"

# Reproducibility, fix seeds
pl.seed_everything(seed, workers=True)
torch.manual_seed(seed)
np.random.seed(seed)

# ============================================================
# Configure accelerator (GOU, CPU, MPS)
# ============================================================

if torch.cuda.is_available():
    accelerator = "gpu"
    devices = 1
    num_workers = num_workers_cuda

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()

    print("Using GPU:", torch.cuda.get_device_name(0))

elif torch.backends.mps.is_available():
    accelerator = "mps"
    devices = 1
    num_workers = 0
    torch.set_float32_matmul_precision("medium")

    print("Using Apple Silicon MPS.")

else:
    accelerator = "cpu"
    devices = 1
    num_workers = 0

    print("Using CPU.")


# ============================================================
# Load dataset and select noise fractions
# ============================================================

dataset = SPECTDataset(
    h5_file=data_file,
    context_size=context_size,
)

def noise_group_name(fraction):
    return f"fraction_{fraction:.4f}".replace(".", "p")


requested_groups = {
    noise_group_name(fraction) for fraction in noise_fractions
}

available_groups = {
    noise_group for _, noise_group, _ in dataset.samples
}

missing_groups = requested_groups - available_groups

if missing_groups:
    raise ValueError(
        f"Requested noise groups are missing from the dataset: "
        f"{sorted(missing_groups)}. Available groups: "
        f"{sorted(available_groups)}"
    )

# Retain only samples from the requested noise fractions.
# In particular, this excludes fraction_1p0000, which is not
# a denoising example because its input is not downsampled.
selected_indices = [
    i
    for i, (_, noise_group, _) in enumerate(dataset.samples)
    if noise_group in requested_groups
]

# Split by scan, not by individual projection.
# This prevents projections from the same scan appearing in
# both training and evaluation sets.
scan_ids = sorted({
    dataset.samples[i][0] for i in selected_indices
})

rng = np.random.default_rng(seed)
rng.shuffle(scan_ids)

n_scans = len(scan_ids)
n_test_scans = max(1, int(round(test_fraction * n_scans)))
n_val_scans = max(1, int(round(val_fraction * n_scans)))

test_scans = set(scan_ids[:n_test_scans])
val_scans = set(scan_ids[n_test_scans:n_test_scans + n_val_scans])
train_scans = set(scan_ids[n_test_scans + n_val_scans:])

train_indices = [
    i for i in selected_indices
    if dataset.samples[i][0] in train_scans
]
val_indices = [
    i for i in selected_indices
    if dataset.samples[i][0] in val_scans
]
test_indices = [
    i for i in selected_indices
    if dataset.samples[i][0] in test_scans
]

if max_train_samples is not None:
    rng.shuffle(train_indices)
    train_indices = train_indices[:max_train_samples]

if max_test_samples is not None:
    rng.shuffle(test_indices)
    test_indices = test_indices[:max_test_samples]

if max_val_samples is not None:
    rng.shuffle(val_indices)
    val_indices = val_indices[:max_val_samples]

train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)
test_data = Subset(dataset, test_indices)

print("\nDataset split")
print(f"Total scans:      {n_scans}")
print(f"Training scans:   {len(train_scans)}")
print(f"Validation scans: {len(val_scans)}")
print(f"Test scans:       {len(test_scans)}")
print(f"Training samples:   {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples:       {len(test_data)}")
print(f"Noise fractions: {noise_fractions}")


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(accelerator == "gpu"),
    persistent_workers=(num_workers > 0),
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(accelerator == "gpu"),
    persistent_workers=(num_workers > 0),
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(accelerator == "gpu"),
    persistent_workers=(num_workers > 0),
)


# ============================================================
# Configure model
# ============================================================

model = SPECTUNet(
    context_size=context_size,
    base_channels=base_channels,
    n_blocks=n_blocks,
    n_res_blocks=n_res_blocks,
    dropout=dropout,
    lr=lr,
)

print("\nModel configuration")
print(f"Context size:  {context_size}")
print(f"U-Net blocks:  {n_blocks}")
print(f"Base channels: {base_channels}")
print(f"Learning rate: {lr}")
print(f"Parameters:    {sum(p.numel() for p in model.parameters()):,}\n\n")


# ============================================================
# Configure Trainer
# ============================================================

loss_tracker = LossTracker()

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename=model_name + "-{epoch:03d}-{val_loss:.5f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=15,
)

trainer = pl.Trainer(
    max_epochs=n_epochs,
    accelerator=accelerator,
    devices=devices,
    strategy="auto",
    callbacks=[
        loss_tracker,
        checkpoint_callback,
        early_stopping,
    ],
    enable_progress_bar=True,
    log_every_n_steps=10,
    check_val_every_n_epoch=1,
    num_sanity_val_steps=0,
    default_root_dir=str(output_dir),
)

# ============================================================
# Train
# ============================================================

if doTraining:
  
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # ============================================================
    # Save model
    # ============================================================

    if checkpoint_callback.best_model_path:
        best_checkpoint = torch.load(
            checkpoint_callback.best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["state_dict"])

    model.export_to_state_dict(str(state_dict_path))

    print(f"\nSaved state dict to: {state_dict_path}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score}")

    # ============================================================
    # Plot training and validation losses
    # ============================================================

    plt.figure(figsize=(10, 7))

    plt.plot(
        loss_tracker.train_losses,
        label="Training loss",
        color="royalblue"
    )

    plt.plot(
        loss_tracker.val_losses,
        label="Validation loss",
        color="firebrick"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 loss in log-count space")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    loss_plot_path = output_dir / f"{model_name}_loss.png"
    plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
    print(f"Loss plot saved to: {loss_plot_path}")


# ============================================================
# Evaluate on the held-out test set
# ============================================================

#load model
best_model = SPECTUNet.load_from_state_dict(
    str(state_dict_path),
    context_size=context_size,
    base_channels=base_channels,
    n_blocks=n_blocks,
    n_res_blocks=n_res_blocks,
    dropout=dropout,
    lr=lr,
)

test_results = trainer.test(
    best_model,
    dataloaders=test_loader,
    verbose=True,
)

print("\nHeld-out test results:")
print(test_results)

# ============================================================
# Plot one held-out test example
# ============================================================

# Select a random sample from the test set.
test_rng = np.random.default_rng(seed + 1)
sample_index = int(test_rng.integers(len(test_indices)))
dataset_index = test_indices[sample_index]

x, y = dataset[dataset_index]

# x: (context_size, rows, cols)
# y: (1, rows, cols)
central_noisy = x[context_size // 2].numpy()
true_clean = y[0].numpy()

device = trainer.strategy.root_device
best_model = best_model.to(device)
best_model.eval()

with torch.no_grad():
    prediction = best_model(x.unsqueeze(0).to(device))[0, 0]
    prediction = prediction.cpu().numpy()

# Use the same colour scale for all three panels.
vmax = max(
    float(np.percentile(central_noisy, 99.5)),
    float(np.percentile(true_clean, 99.5)),
    float(np.percentile(prediction, 99.5)),
)
vmax = max(vmax, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

images = [
    (central_noisy, "Noisy input"),
    (prediction, "Predicted clean projection"),
    (true_clean, "True clean projection"),
]

for ax, (image, title) in zip(axes, images):
    im = ax.imshow(
        image,
        cmap="viridis",
        origin="upper",
        #vmin=0,            #fix scale for each plot?
        #vmax=vmax,         #fix scale for each plot?
    )
    ax.set_title(title)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

scan_id, noise_group, angle_idx = dataset.samples[dataset_index]

fig.suptitle(
    f"Test example: {scan_id}, angle {angle_idx}, {noise_group}"
)

plt.tight_layout()

comparison_path = output_dir / f"{model_name}_test_comparison.png"
plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
#plt.show()

print(f"Test comparison saved to: {comparison_path}")
