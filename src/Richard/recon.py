
import os
import time

import numpy as np
import pydicom
import torch
import matplotlib.pyplot as plt

from scipy.ndimage import binary_closing, binary_fill_holes

from pytomography.io.SPECT import dicom
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM


# ============================================================
# SETTINGS
# ============================================================

file_NM = (
    "PPMI_1/100001/DaT_Brain_Tomo/2020-09-09_17_07_33.0/"
    "I1449481/"
    "PPMI_100001_NM_DaT_Brain_Tomo_br_raw_20210601144217264_1_S1028880_I1449481.dcm"
)

output_dir = "reconstruction"
os.makedirs(output_dir, exist_ok=True)

N_ITERS = 2
N_SUBSETS = 8

MU = 0.14             # Attenuation coefficient, cm^-1
MU_THRESHOLD = 0.05    # Relative threshold for head mask


# ============================================================
# LOAD PROJECTIONS
# ============================================================

start = time.time()

ds = pydicom.dcmread(file_NM)

object_meta, proj_meta = dicom.get_metadata(
    file_NM,
    index_peak=0,
)

projections = dicom.get_projections(
    file_NM,
    index_peak=0,
)

print("Projection shape:", tuple(projections.shape))
print("Total counts:", projections.sum().item())
print("Object shape:", object_meta.shape)
print("Voxel size (cm):", object_meta.dr)


# ============================================================
# OPTIONAL: REDUCE COUNTS THROUGH POISSON / BINOMIAL SAMPLING
# ============================================================

# fraction = 0.5
# torch.manual_seed(12345)
# projections = torch.binomial(
#     torch.round(projections).float(),
#     torch.full_like(projections, fraction),
# )
# print("Downsampled counts:", projections.sum().item())


# ============================================================
# PRELIMINARY OSEM RECONSTRUCTION
# Used to estimate the head mask for attenuation correction.
# ============================================================

system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)

likelihood = PoissonLogLikelihood(
    system_matrix,
    projections,
)

preliminary = OSEM(likelihood)(
    n_iters=N_ITERS,
    n_subsets=N_SUBSETS,
)

# Estimate the head outline from the preliminary reconstruction.
recon_np = preliminary.detach().cpu().numpy()

mask = recon_np > MU_THRESHOLD * recon_np.max()

# Collapse through the axial direction, clean the outline,
# then extend it through the volume.
mask_2d = mask.any(axis=2)
mask_2d = binary_closing(mask_2d, iterations=2)
mask_2d = binary_fill_holes(mask_2d)

mask_3d = np.repeat(
    mask_2d[:, :, None],
    recon_np.shape[2],
    axis=2,
)

# Uniform attenuation map: zero outside the head,
# MU inside the head.
mu_map = torch.from_numpy(
    (mask_3d * MU).astype(np.float32)
)

print("Attenuation map shape:", tuple(mu_map.shape))


# ============================================================
# OSEM WITH ATTENUATION CORRECTION
# ============================================================

attenuation = SPECTAttenuationTransform(
    attenuation_map=mu_map,
)
attenuation.configure(object_meta, proj_meta)

transforms = [attenuation]

# ============================================================
# OPTIONAL: INCLUDE A COLLIMATOR PSF MODEL
# Uncomment to enable. Choose the appropriate collimator
# model for your scanner from "SY-ME" or "SY-LEHR".
# ============================================================

# from pytomography.transforms.SPECT import SPECTPSFTransform
#
# energy = 0.5 * (
#     float(ds.EnergyWindowInformationSequence[0]
#           .EnergyWindowRangeSequence[0].EnergyWindowLowerLimit)
#     + float(ds.EnergyWindowInformationSequence[0]
#             .EnergyWindowRangeSequence[0].EnergyWindowUpperLimit)
# )
#
# psf_meta = dicom.get_psfmeta_from_scanner_params(
#     collimator_name="SY-LEHR",
#     energy_keV=energy,
# )
#
# psf = SPECTPSFTransform(psf_meta)
# transforms.append(psf)


system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=transforms,
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)

likelihood = PoissonLogLikelihood(
    system_matrix,
    projections,
)

reconstruction = OSEM(likelihood)(
    n_iters=N_ITERS,
    n_subsets=N_SUBSETS,
)

print("Reconstruction range:",
      reconstruction.min().item(),
      reconstruction.max().item())

# ============================================================
# OPTIONAL: SAVE RECONSTRUCTION
# ============================================================

# torch.save(
#     reconstruction.cpu(),
#     os.path.join(output_dir, "OSEM_attenuation.pt"),
# )


# ============================================================
# PLOT SLICES, SUMMED SLICES, MAX OF SLICES
# ============================================================

recon = reconstruction.detach().cpu().numpy()

cx, cy, cz = (
    recon.shape[0] // 2,
    recon.shape[1] // 2,
    21,
)

images = [
    recon[cx, :, :].T,
    recon[:, cy, :].T,
    recon[:, :, cz].T,

    recon.sum(axis=0).T,
    recon.sum(axis=1).T,
    recon.sum(axis=2).T,

    recon.max(axis=0).T,
    recon.max(axis=1).T,
    recon.max(axis=2).T,
]

titles = [
    f"Sagittal Slice {cx}",
    f"Coronal Slice {cy}",
    f"Axial Slice {cz}",

    "Sagittal integrated",
    "Coronal integrated",
    "Axial integrated",

    "Sagittal maximum",
    "Coronal maximum",
    "Axial maximum",
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, image, title in zip(axes.flat, images, titles):
    ax.imshow(image, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "OSEM_attenuation.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.show()


print(f"Total runtime: {time.time() - start:.1f} s")
