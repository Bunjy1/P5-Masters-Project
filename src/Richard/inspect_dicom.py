import pydicom
import matplotlib.pyplot as plt
import numpy as np

def load_spect_dicom(filename):

    ds = pydicom.dcmread(filename)

    projections = ds.pixel_array.astype(np.float32)

    # Split the two detector heads
    projections_det1 = projections[:120]
    projections_det2 = projections[120:]

    # Acquisition angles
    rotation = ds.RotationInformationSequence[0]

    detector1 = ds.DetectorInformationSequence[0]
    detector2 = ds.DetectorInformationSequence[1]

    angles_det1 = (
        float(detector1.StartAngle)
        + np.arange(120) * float(rotation.AngularStep)
    )

    angles_det2 = (
        float(detector2.StartAngle)
        + np.arange(120) * float(rotation.AngularStep)
    )

    return {
        "ds": ds,

        "projections": projections,

        "detector1": projections_det1,
        "detector2": projections_det2,

        "angles_det1": angles_det1,
        "angles_det2": angles_det2,

        "pixel_size": np.asarray(ds.PixelSpacing, dtype=float),

        "radius_det1": np.asarray(
            detector1.RadialPosition, dtype=float
        ),

        "radius_det2": np.asarray(
            detector2.RadialPosition, dtype=float
        ),
    }

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

filename = "PPMI_1/101747/DaT_Brain_Tomo/2021-05-18_15_34_42.0/I1472297/PPMI_101747_NM_DaT_Brain_Tomo_br_raw_20210722155928010_1_S1046871_I1472297.dcm"

# ------------------------------------------------------------
# Read DICOM
# ------------------------------------------------------------

ds = pydicom.dcmread(filename)

# to print entire dicom dataset
print("\n============================================================")
print("DICOM DATASET")
print("============================================================\n")

print(ds)

print("\n============================================================")
print("SPECT DICOM SUMMARY")
print("============================================================\n")

# ------------------------------------------------------------
# General
# ------------------------------------------------------------

print("GENERAL")
print(f"  Modality              : {getattr(ds, 'Modality', 'N/A')}")
print(f"  Manufacturer          : {getattr(ds, 'Manufacturer', 'N/A')}")
print(f"  Model                 : {getattr(ds, 'ManufacturerModelName', 'N/A')}")
print(f"  Series description    : {getattr(ds, 'SeriesDescription', 'N/A')}")
print(f"  Image type            : {getattr(ds, 'ImageType', 'N/A')}")
print()

# ------------------------------------------------------------
# Image dimensions
# ------------------------------------------------------------

print("IMAGE")
print(f"  Matrix                : "
      f"{getattr(ds, 'Rows', 'N/A')} x "
      f"{getattr(ds, 'Columns', 'N/A')}")

print(f"  Number of frames      : {getattr(ds, 'NumberOfFrames', 'N/A')}")
print(f"  Pixel spacing (mm)    : {getattr(ds, 'PixelSpacing', 'N/A')}")
print(f"  Bits allocated        : {getattr(ds, 'BitsAllocated', 'N/A')}")
print(f"  Bits stored           : {getattr(ds, 'BitsStored', 'N/A')}")
print(f"  Photometric           : {getattr(ds, 'PhotometricInterpretation', 'N/A')}")
print()

# ------------------------------------------------------------
# Energy window
# ------------------------------------------------------------

print("ENERGY WINDOW")

print(f"  Number of windows     : "
      f"{getattr(ds, 'NumberOfEnergyWindows', 'N/A')}")

if hasattr(ds, "EnergyWindowInformationSequence"):
    for i, window in enumerate(ds.EnergyWindowInformationSequence):
        if hasattr(window, "EnergyWindowRangeSequence"):
            energy = window.EnergyWindowRangeSequence[0]

            print(f"  Window {i + 1}:")
            print(f"    Lower limit (keV)   : "
                  f"{getattr(energy, 'EnergyWindowLowerLimit', 'N/A')}")
            print(f"    Upper limit (keV)   : "
                  f"{getattr(energy, 'EnergyWindowUpperLimit', 'N/A')}")

print()

# ------------------------------------------------------------
# Detectors / collimator
# ------------------------------------------------------------

print("DETECTORS")

print(f"  Number of detectors   : "
      f"{getattr(ds, 'NumberOfDetectors', 'N/A')}")

if hasattr(ds, "DetectorInformationSequence"):

    for i, detector in enumerate(ds.DetectorInformationSequence):

        print(f"\n  Detector {i + 1}")

        print(f"    Collimator          : "
              f"{getattr(detector, 'CollimatorGridName', 'N/A')}")

        print(f"    Collimator type     : "
              f"{getattr(detector, 'CollimatorType', 'N/A')}")

        print(f"    Focal distance      : "
              f"{getattr(detector, 'FocalDistance', 'N/A')}")

        print(f"    Field of view      : "
              f"{getattr(detector, 'FieldOfViewDimension', 'N/A')}")

        print(f"    Zoom factor         : "
              f"{getattr(detector, 'ZoomFactor', 'N/A')}")

        print(f"    Start angle         : "
              f"{getattr(detector, 'StartAngle', 'N/A')}")

        if hasattr(detector, "RadialPosition"):
            radial = detector.RadialPosition

            print(f"    Radial positions    : "
                  f"{len(radial)} values")

            print(f"    Radius range (mm)   : "
                  f"{min(radial):.2f} -> {max(radial):.2f}")

# ------------------------------------------------------------
# Rotation / acquisition
# ------------------------------------------------------------

print("\nROTATION / ACQUISITION")

if hasattr(ds, "RotationInformationSequence"):

    rotation = ds.RotationInformationSequence[0]

    print(f"  Rotation direction    : "
          f"{getattr(rotation, 'RotationDirection', 'N/A')}")

    print(f"  Scan arc (degrees)    : "
          f"{getattr(rotation, 'ScanArc', 'N/A')}")

    print(f"  Angular step          : "
          f"{getattr(rotation, 'AngularStep', 'N/A')}")

    print(f"  Frames / rotation     : "
          f"{getattr(rotation, 'NumberOfFramesInRotation', 'N/A')}")

    print(f"  Start angle           : "
          f"{getattr(rotation, 'StartAngle', 'N/A')}")

    print(f"  Frame duration (ms)   : "
          f"{getattr(rotation, 'ActualFrameDuration', 'N/A')}")

# ------------------------------------------------------------
# Pixel data
# ------------------------------------------------------------

print("\nPIXEL DATA")

print(f"  PixelData present     : {'PixelData' in ds}")

if hasattr(ds, "PixelData"):
    print(f"  PixelData length      : {len(ds.PixelData):,} bytes")

print("\n============================================================")

# Load all projection frames
images = ds.pixel_array

print("Pixel array:")
print("  shape:", images.shape)
print("  dtype:", images.dtype)
print()


print("Detector vector:")
print(ds.DetectorVector)

print("\nAngular view vector:")
print(ds.AngularViewVector)

print("\nRotation vector:")
print(ds.RotationVector)

print("Frame information:")
print()

for i in range(min(20, len(images))):

    print(
        f"Frame {i:3d} | "
        f"Detector {ds.DetectorVector[i]:2d} | "
        f"Angle {ds.AngularViewVector[i]:8.3f}"
    )

data = load_spect_dicom(filename)

print(data["detector1"].shape)
print(data["detector2"].shape)

print(data["angles_det1"])
print(data["angles_det2"])


frames = [0, 1, 2, 3, 118, 119, 120, 121]


# ------------------------------------------------------------
# Plot a few frames
# ------------------------------------------------------------


fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for ax, frame in zip(axes.flat, frames):

    ax.imshow(images[frame], cmap="gray")
    ax.set_title(
        f"Frame {frame}\n"
        f"det={ds.DetectorVector[frame]}, "
        f"angle={ds.AngularViewVector[frame]:.1f}°"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()