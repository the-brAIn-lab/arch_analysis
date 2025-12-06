# furthest_sum_gpu.py
from __future__ import annotations
from typing import Iterable, List, Optional
import numpy as np

try:
    import cupy as cp
except Exception as e:
    cp = None  # Optional: allow importing the module without CuPy


def furthest_sum_gpu(
    K,
    noc: int,
    i: int | Iterable[int],
    exclude: Optional[Iterable[int]] = None,
    *,
    treat_as_kernel: Optional[bool] = None,
    one_based: bool = False,
    device: int | None = None,
) -> List[int]:
    """
    GPU version of FurthestSum using CuPy (NVIDIA).

    Parameters
    ----------
    K : array-like (CuPy or NumPy)
        - Data matrix of shape (D, N) with observations as **columns**, or
        - Kernel/Gram matrix of shape (N, N) (symmetric).
        If NumPy is provided, it is copied to GPU.
    noc : int
        Number of candidate archetypes to extract.
    i : int or iterable of int
        Initial seed(s). First element is the rolling index.
    exclude : iterable of int, optional
        Indices that cannot be selected.
    treat_as_kernel : bool, optional
        If None, auto-detect: square & symmetric => kernel; else data.
    one_based : bool, default False
        Interpret input/output indices as 1-based (MATLAB-style).
    device : int or None
        CUDA device id to use. If None, current device is used.

    Returns
    -------
    selected : list[int]
        Selected indices (0-based by default; 1-based if `one_based=True`).

    Notes
    -----
    - Data mode (D x N): distances via dot-products:
        sqrt(||x_j||^2 - 2 <x_s, x_j> + ||x_s||^2).
    - Kernel mode (N x N): distances via kernel trick:
        sqrt(K_jj - 2 K_js + K_ss).
    - Follows MATLAB's rolling schedule (noc+10 iterations, drop earliest after noc-1).
    """
    if cp is None:
        raise ImportError(
            "CuPy is required for furthest_sum_gpu. Install with `pip install cupy-cuda12x` "
            "matching your CUDA runtime."
        )

    # Device context (optional)
    if device is not None:
        dev = cp.cuda.Device(device)
        dev.use()

    # Move to GPU if needed
    A = cp.asarray(K)

    if A.ndim != 2:
        raise ValueError("K must be a 2D array.")

    # Auto-detect kernel/data if not specified
    if treat_as_kernel is None:
        is_square = A.shape[0] == A.shape[1]
        if is_square:
            # symmetry check on GPU
            treat_as_kernel = bool(cp.allclose(A, A.T, atol=1e-10))
        else:
            treat_as_kernel = False

    # N = number of observations (columns for data; rows for kernel)
    N = int(A.shape[0] if treat_as_kernel else A.shape[1])

    # Helpers to translate indices
    def to0(idx: Iterable[int] | int) -> List[int]:
        if isinstance(idx, (int, np.integer)):
            return [int(idx) - 1 if one_based else int(idx)]
        return [int(x) - 1 if one_based else int(x) for x in idx]

    selected: List[int] = to0(i)
    if len(selected) == 0:
        raise ValueError("Initial seed `i` must contain at least one index.")
    rolling_idx = selected[0]

    # Candidate mask: True => available, False => banned
    available = cp.ones(N, dtype=cp.bool_)
    if exclude is not None:
        ex = cp.asarray(to0(exclude))
        available[ex] = False
    if selected:
        available[cp.asarray(selected)] = False

    # Cumulative distances (on GPU)
    sum_dist = cp.zeros(N, dtype=A.dtype)

    # Branch-specific distance helpers
    if treat_as_kernel:
        # Kernel mode
        Kdiag = cp.diag(A).astype(A.dtype)

        def add_from(seed: int):
            # sqrt(max(Kjj - 2*Kjs + Kss, 0))
            d2 = Kdiag - 2.0 * A[seed, :] + Kdiag[seed]
            cp.maximum(d2, 0.0, out=d2)
            return cp.sqrt(d2)

        remove_from = add_from  # same magnitude; we subtract it

    else:
        # Data mode
        X = A  # (D, N), columns are observations
        norms2 = cp.sum(X * X, axis=0)

        def add_from(seed: int):
            # <x_s, X> for all columns
            Kq = X[:, seed].T @ X  # shape (N,)
            d2 = norms2 - 2.0 * Kq + norms2[seed]
            cp.maximum(d2, 0.0, out=d2)
            return cp.sqrt(d2)

        remove_from = add_from

    # Main rolling loop
    # Matches MATLAB: for k=1:noc+10; if k>noc-1 remove earliest; then add seed distances; pick farthest
    for k in range(1, noc + 10 + 1):
        if k > (noc - 1) and len(selected) > 0:
            to_remove = selected[0]
            sum_dist -= remove_from(to_remove)
            available[to_remove] = True  # make available again
            selected.pop(0)

        sum_dist += add_from(rolling_idx)

        # Pick farthest among available
        cand_idx = cp.nonzero(available)[0]
        if cand_idx.size == 0:
            break
        # argmax over candidates
        local_arg = cp.argmax(sum_dist[cand_idx])
        farthest_idx = int(cand_idx[local_arg].get())  # bring scalar to host

        rolling_idx = farthest_idx
        selected.append(farthest_idx)
        available[farthest_idx] = False

    # Ensure exactly noc outputs
    if len(selected) > noc:
        selected = selected[-noc:]
    elif len(selected) < noc:
        # pad with remaining far candidates if degenerate
        mask = cp.ones(N, dtype=cp.bool_)
        mask[cp.asarray(selected)] = False
        rest = cp.nonzero(mask)[0].get().tolist()
        need = noc - len(selected)
        selected.extend(rest[:need])

    if one_based:
        return [s + 1 for s in selected]
    return selected
