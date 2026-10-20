import matplotlib.pyplot as plt
import numpy as np
import torch

from SPECTDataset import SPECTDataset

# Use the dataset directly so we can inspect individual samples.
dataset = SPECTDataset(
    h5_file="spect_dataset.h5",
    context_size=5,
)

fig, axes = plt.subplots(3, 6, figsize=(18, 9))

for row, idx in enumerate([0, len(dataset) // 2, len(dataset) - 1]):
    x, y = dataset[idx]

    # x: (5, rows, cols)
    # y: (1, rows, cols)
    print(
        f"Sample {idx}: "
        f"x shape = {tuple(x.shape)}, "
        f"y shape = {tuple(y.shape)}, "
        f"x counts = {x.sum().item():.0f}, "
        f"y counts = {y.sum().item():.0f}"
    )

    # Display the five noisy context projections.
    for j in range(x.shape[0]):
        ax = axes[row, j]
        image = x[j].numpy()
        ax.imshow(image, origin="lower", cmap="magma", vmin=0)
        ax.set_title(f"Noisy context {j - x.shape[0] // 2:+d}")
        ax.axis("off")

    # Display the clean target.
    ax = axes[row, 5]
    target = y[0].numpy()
    ax.imshow(target, origin="lower", cmap="magma", vmin=0)
    ax.set_title("Clean target")
    ax.axis("off")

plt.tight_layout()
plt.show()