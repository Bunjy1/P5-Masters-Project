
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SPECTDataset(Dataset):
    def __init__(self, h5_file, context_size=5):
        """
        Args:
            h5_file: Path to the HDF5 dataset.
            context_size: Number of neighbouring projections supplied
                          to the model. Must be a positive odd integer.
        """
        if context_size < 1 or context_size % 2 == 0:
            raise ValueError("context_size must be a positive odd integer.")

        self.h5_file = str(h5_file)
        self.context_size = context_size
        self.half_context = context_size // 2

        # Build a lightweight index without keeping the file open.
        self.samples = []

        with h5py.File(self.h5_file, "r") as h5:
            scan_ids = sorted(h5["scans"].keys())
            noise_groups = sorted(h5["noisy"].keys())

            for scan_id in scan_ids:
                n_angles = h5["scans"][scan_id].attrs["n_angles"]

                # Valid central angles with a complete context.
                for angle_idx in range(
                    self.half_context,
                    n_angles - self.half_context,
                ):
                    for noise_group in noise_groups:
                        self.samples.append(
                            (scan_id, noise_group, angle_idx)
                        )

        self._h5 = None

        print(f"Loaded index with {len(self.samples)} samples.")
        print(f"Context size: {self.context_size}")
        print(f"Noise levels: {len(noise_groups)}")

    def __len__(self):
        return len(self.samples)

    def _get_file(self):
        # Open lazily, separately in each DataLoader worker.
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r")
        return self._h5

    def __getitem__(self, idx):
        h5 = self._get_file()

        scan_id, noise_group, angle_idx = self.samples[idx]

        start = angle_idx - self.half_context
        stop = angle_idx + self.half_context + 1

        # Noisy neighbouring projections, including the centre.
        x = h5["noisy"][noise_group][scan_id][start:stop]

        # Clean central projection: always the corresponding target.
        y = h5["scans"][scan_id]["projections"][angle_idx]

        # Copy to ensure independent, writable NumPy arrays.
        x = np.asarray(x, dtype=np.float32).copy()
        y = np.asarray(y, dtype=np.float32).copy()

        x = torch.from_numpy(x)       # (context_size, rows, cols)
        y = torch.from_numpy(y).unsqueeze(0)  # (1, rows, cols)

        return x, y

    def __getstate__(self):
        # Do not pickle an open HDF5 handle into DataLoader workers.
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()
