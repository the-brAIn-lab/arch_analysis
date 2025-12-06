# furthest_sum.py
from __future__ import annotations
import numpy as np
from typing import Iterable, List, Optional, Tuple

def furthest_sum(
    K: np.ndarray,
    noc: int,
    i: int | Iterable[int],
    exclude: Optional[Iterable[int]] = None,
    *,
    treat_as_kernel: Optional[bool] = None,
    one_based: bool = False,
) -> List[int]:
    """
    Python equivalent of FurthestSum.m

    Parameters
    ----------
    K : ndarray
        Either:
          - data matrix of shape (D, N) with N observations as **columns**, or
          - kernel/Gram matrix of shape (N, N) (symmetric, PSD).
    noc : int
        Number of candidate archetypes to extract.
    i : int or iterable of int
        Initial seed index/indices. If multiple are given, the first is used
        as the rolling index, and all are included in the output set.
    exclude : iterable of int, optional
        Indices that cannot be selected.
    treat_as_kernel : bool, optional
        Force interpretation of K. If None (default), auto-detect:
        square & symmetric => kernel; else => data matrix.
    one_based : bool, default False
        If True, input/output indices are 1-based (MATLAB-style). Internally
        converted to 0-based.

    Returns
    -------
    selected : list[int]
        Selected indices (0-based by default; 1-based if `one_based=True`).

    Notes
    -----
    - Data mode (D x N): distances computed efficiently via dot-products:
        ||x_j - x_s|| = sqrt(||x_j||^2 - 2 <x_j, x_s> + ||x_s||^2).
    - Kernel mode (N x N): distances via the kernel trick:
        ||x_j - x_s|| = sqrt(K_jj - 2 K_js + K_ss).
    - Matches MATLAB’s rolling scheme:
        for k = 1 .. noc+10:
            if k > noc-1: remove the earliest picked index from the active set
            pick new index that maximizes cumulative distance to the (rolling) set
    """
    A = np.asarray(K)
    if A.ndim != 2:
        raise ValueError("K must be 2D.")

    if treat_as_kernel is None:
        treat_as_kernel = (A.shape[0] == A.shape[1]) and np.allclose(A, A.T, atol=1e-10)

    if treat_as_kernel:
        N = A.shape[0]
    else:
        # Data matrix: D x N (observations are columns)
        if A.shape[0] < A.shape[1]:
            # Common case: many observations
            N = A.shape[1]
        else:
            # If rows >= cols, still treat columns as observations
            N = A.shape[1]

    # Prepare indices
    def to0(idx: Iterable[int] | int) -> List[int]:
        if isinstance(idx, (int, np.integer)):
            return [int(idx) - 1 if one_based else int(idx)]
        return [int(x) - 1 if one_based else int(x) for x in idx]

    selected: List[int] = to0(i)
    if len(selected) == 0:
        raise ValueError("Initial seed `i` must contain at least one index.")
    rolling_idx = selected[0]

    banned = np.zeros(N, dtype=bool)
    if exclude is not None:
        banned[to0(exclude)] = True
    banned[np.clip(selected, 0, N - 1)] = True  # do not reselect current picks

    # Cumulative distances to the (rolling) set
    sum_dist = np.zeros(N, dtype=float)

    if treat_as_kernel:
        # Kernel distances: d(j, s) = sqrt(K_jj - 2*K_js + K_ss)
        Kdiag = np.diag(A).astype(float)
        # Helper to add/remove distances to/from sum_dist
        def add_from(seed: int):
            d2 = Kdiag - 2.0 * A[seed, :] + Kdiag[seed]
            np.maximum(d2, 0.0, out=d2)
            np.sqrt(d2, out=d2)
            return d2

        def remove_from(seed: int):
            return add_from(seed)  # same magnitude, we subtract it

    else:
        # Data distances via dot products
        # Precompute squared norms of columns
        X = A  # D x N
        norms2 = np.sum(X * X, axis=0)

        def add_from(seed: int):
            # <x_s, X> for all columns
            Kq = np.dot(X[:, seed].T, X)  # shape (N,)
            d2 = norms2 - 2.0 * Kq + norms2[seed]
            np.maximum(d2, 0.0, out=d2)
            np.sqrt(d2, out=d2)
            return d2

        def remove_from(seed: int):
            return add_from(seed)

    # Main loop (noc + 10 iterations with rolling removal after noc-1)
    # This yields exactly `noc` kept indices in the end (matching MATLAB).
    for k in range(1, noc + 10 + 1):
        # After a short burn-in, drop the earliest selected index from the active set
        if k > (noc - 1) and len(selected) > 0:
            to_remove = selected[0]
            sum_dist -= remove_from(to_remove)
            banned[to_remove] = False  # make it available again
            selected.pop(0)            # remove earliest

        # Update cumulative distances wrt current rolling index
        sum_dist += add_from(rolling_idx)

        # Choose the farthest allowed index
        candidates = np.where(~banned)[0]
        if candidates.size == 0:
            # No feasible candidates; break
            break
        farthest_idx = candidates[np.argmax(sum_dist[candidates])]

        # Update state
        rolling_idx = farthest_idx
        selected.append(farthest_idx)
        banned[farthest_idx] = True

    # Return exactly `noc` indices (MATLAB ends with noc due to add/remove schedule)
    if len(selected) > noc:
        selected = selected[-noc:]
    elif len(selected) < noc:
        # In degenerate cases, pad by unique far candidates
        fill_needed = noc - len(selected)
        more = np.setdiff1d(np.arange(N), np.array(selected), assume_unique=False)
        selected.extend(list(more[:fill_needed]))

    if one_based:
        return [s + 1 for s in selected]
    return selected
