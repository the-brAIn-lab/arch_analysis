# generate_synthetic_noise.py
from __future__ import annotations
import numpy as np
from typing import Iterable, Optional, Sequence, Tuple

def generate_synthetic_noise(
    sx: int,
    sy: int,
    noise_var: Optional[Sequence[float]] = None,
    stepsize: int = 8,
    show_plot: bool = False,
) -> np.ndarray:
    """
    Python equivalent of generateSyntheticNoise.m

    Parameters
    ----------
    sx, sy : int
        Map size in x (rows) and y (cols).
    noise_var : sequence of float, optional
        Max variances for each noise map. Defaults to [1, 4, 16].
    stepsize : int, default 8
        Radial step size for the nested circular contours (in pixels).
    show_plot : bool, default False
        If True, shows three rows of plots for each variance:
        (variance map, std-dev map, one sample N(0, var(x,y))).

    Returns
    -------
    noise : ndarray, shape (sx, sy, K)
        Stack of variance maps, one per entry in `noise_var`.
    """
    if noise_var is None:
        noise_var = np.array([1, 2, 4], dtype=float) ** 2
    else:
        noise_var = np.asarray(noise_var, dtype=float)

    # Center and max radius
    cx = sx / 2.0 + 0.5
    cy = sy / 2.0 + 0.5
    rx = sx / 2.0
    # ry = sy / 2.0  # kept for parity; only circular contours are used

    num_noise = noise_var.size
    noise = np.zeros((sx, sy, num_noise), dtype=float)

    # Proper 2D grid (MATLAB fix: their code used meshgrid(1:sx) which only works if sx == sy)
    rr, cc = np.meshgrid(
        np.arange(1, sx + 1, dtype=float),
        np.arange(1, sy + 1, dtype=float),
        indexing="ij",
    )
    dist = np.sqrt((rr - cx) ** 2 + (cc - cy) ** 2)

    # Radii from outside to inside
    noise_contours = np.arange(rx, 0, -stepsize)  # [rx, rx-stepsize, ..., >0]

    for j in range(num_noise):
        X = np.zeros((sx, sy), dtype=float)
        # Linearly ramp variance from 1% of max up to max across the contours
        var_levels = np.linspace(0.01 * noise_var[j], noise_var[j], num=noise_contours.size)

        for r, v in zip(noise_contours, var_levels):
            mask = dist <= r
            X[mask] = v

        noise[:, :, j] = X

    if show_plot:
        import matplotlib.pyplot as plt
        import numpy.ma as ma

        ncols = num_noise
        fig, axes = plt.subplots(3, ncols, figsize=(3.8 * ncols, 10), constrained_layout=True)

        # Ensure axes is 2D even for ncols=1
        if ncols == 1:
            axes = np.asarray(axes).reshape(3, 1)

        vmax = float(np.max(noise_var))
        for j in range(num_noise):
            X = noise[:, :, j]
            Xstd = np.sqrt(X)
            sample = np.random.randn(sx, sy) * Xstd

            # Hide zeros by masking (parity with MATLAB's X(X==0)=nan before pcolor)
            mX = ma.masked_where(X == 0, X)
            mXstd = ma.masked_where(Xstd == 0, Xstd)
            mSample = ma.masked_where(X == 0, sample)

            im0 = axes[0, j].imshow(mX, origin="lower", aspect="equal")
            axes[0, j].set_title(f"Max variance {noise_var[j]:.2f}")
            fig.colorbar(im0, ax=axes[0, j])

            im1 = axes[1, j].imshow(mXstd, origin="lower", aspect="equal")
            axes[1, j].set_title(f"Max std dev {np.sqrt(noise_var[j]):.2f}")
            fig.colorbar(im1, ax=axes[1, j])

            im2 = axes[2, j].imshow(mSample, origin="lower", aspect="equal", vmin=0, vmax=vmax)
            axes[2, j].set_title(f"Normal noise N(0,{int(noise_var[j])})")
            fig.colorbar(im2, ax=axes[2, j])

            for row in range(3):
                axes[row, j].set_xlim(0, sy - 1)
                axes[row, j].set_ylim(0, sx - 1)
                axes[row, j].set_xticks([])
                axes[row, j].set_yticks([])

        plt.show()

    return noise
