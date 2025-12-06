# estimate_background_noise.py
from __future__ import annotations
import numpy as np
import nibabel as nib
from typing import Optional

def estimate_background_noise(file: str, file_filt: Optional[str]) -> float:
    """
    Python equivalent of estimateBackgroundNoise.m

    Parameters
    ----------
    file : str
        Path to raw fMRI NIfTI (used to compute background mask).
    file_filt : str or None
        Path to filtered/preprocessed NIfTI (used to compute noise variance).
        If None, uses `file` for both steps (i.e., estimate on the same data).

    Returns
    -------
    noise_threshold : float
        Mean of per-voxel variances (over time) for background voxels ( > 0 ).
    """
    # ---- Load raw (for mask) ----
    img_raw = nib.load(file)
    data_raw = img_raw.get_fdata(dtype=np.float32)  # shape: (X, Y, Z, T)
    if data_raw.ndim != 4:
        raise ValueError(f"`file` must be 4D (got shape {data_raw.shape}).")

    x, y, z, T = data_raw.shape
    V = x * y * z
    X = data_raw.reshape(V, T)

    # MATLAB: not_brain = mean(X') < mean(X(:)) * .8
    # mean(X') in MATLAB == per-voxel mean over time (i.e., mean along axis=1 here)
    voxel_means = X.mean(axis=1)
    global_mean = X.mean()
    not_brain = voxel_means < (0.8 * global_mean)

    # ---- Load filtered (for variance) ----
    if file_filt is None:
        data_filt = data_raw  # estimate on the same data
    else:
        img_filt = nib.load(file_filt)
        data_filt = img_filt.get_fdata(dtype=np.float32)
        if data_filt.shape != (x, y, z, T):
            # We only need that time dimension matches; but for simplicity keep strict parity
            raise ValueError(
                f"`file` and `file_filt` must have same shape. "
                f"Got raw {data_raw.shape} vs filt {data_filt.shape}."
            )

    F = data_filt.reshape(V, T)

    # MATLAB: var(filtered_data(not_brain,: )') 
    # -> per-voxel variance across time, using sample variance (ddof=1)
    if not np.any(not_brain):
        raise ValueError("Background mask is empty; check your data or threshold.")

    var_X = np.var(F[not_brain, :], axis=1, ddof=1)  # per background voxel
    positive = var_X > 0
    if not np.any(positive):
        raise ValueError("All background variances are zero or NaN; check inputs.")

    noise_threshold = float(var_X[positive].mean())
    return noise_threshold
