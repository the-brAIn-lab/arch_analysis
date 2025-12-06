# multi_subject_aa.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time

try:
    import cupy as cp  # optional
except Exception:
    cp = None

from mgetopt import mgetopt
from SupdateIndiStep import supdate_indi_step
from furthest_sum import furthest_sum


@dataclass
class Subject:
    """Container matching the MATLAB 'subj' struct."""
    X: np.ndarray          # shape (T, V)
    sX: np.ndarray         # shape (T, sV)


def multi_subject_aa(
    subj: List[Subject],
    noc: int,
    opts: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, float, float]:
    """
    Python equivalent of MultiSubjectAA.m

    Parameters
    ----------
    subj : list of Subject
        Each Subject has X (T x V) and sX (T x sV).
    noc : int
        Number of archetypes/components.
    opts : dict, optional
        Keys (default):
          - 'maxiter' (100)
          - 'conv_crit' (1e-6)
          - 'fix_var_iter' (5)
          - 'use_gpu' (False)  # CuPy if True
          - 'heteroscedastic' (True)
          - 'numCstep' (10)
          - 'numSstep' (20)
          - 'sort_crit' ('corr' | 'energy')
          - 'init' ('FurthestSum' | 'random')
          - 'initSstep' (250)
          - 'rngSEED' (None)
          - 'noise_threshold' (None -> computed from SST)

    Returns
    -------
    results_subj : list of dict
        For each subject: S, sXC, sigmaSq, NLL, SSE, SST, SST_sigmaSq (all numpy arrays/scalars).
    C : ndarray, shape (sV, noc)
        Archetype generator matrix (columns lie on simplex).
    cost_fun : ndarray, shape (iters,)
        Negative log-likelihood per iteration.
    varexpl : float
        Fraction of variance explained ((SST - sum(SSE))/SST).
    time_taken : float
        Wall-clock seconds.
    """
    if opts is None:
        opts = {}

    conv_crit      = mgetopt(opts, 'conv_crit', 1e-6)
    maxiter        = int(mgetopt(opts, 'maxiter', 100))
    fix_var_iter   = int(mgetopt(opts, 'fix_var_iter', 5))
    runGPU         = bool(mgetopt(opts, 'use_gpu', False))
    voxelVariance  = bool(mgetopt(opts, 'heteroscedastic', True))
    numCstep       = int(mgetopt(opts, 'numCstep', 10))
    numSstep       = int(mgetopt(opts, 'numSstep', 20))
    sort_crit      = mgetopt(opts, 'sort_crit', 'corr')
    init_type      = mgetopt(opts, 'init', 'FurthestSum')
    initial_S_steps= int(mgetopt(opts, 'initSstep', 250))
    rngSEED        = mgetopt(opts, 'rngSEED', None)

    if runGPU and cp is None:
        raise ImportError("opts['use_gpu']=True requested but CuPy is not installed.")

    rng = np.random.default_rng(rngSEED)

    # Shapes (assume all subjects share V and sV)
    V  = subj[0].X.shape[1]   # voxels
    sV = subj[0].sX.shape[1]  # 'tilde' voxels (basis rows for C)
    B  = len(subj)
    T_list = [s.sX.shape[0] for s in subj]  # timesteps per subject

    xp = cp if runGPU else np  # array module

    # ---------- Initialize C ----------
    if init_type.lower() == 'furthestsum':
        # Stack sX across subjects by time (like MATLAB)
        total_T = sum(T_list)
        Xcombined = np.zeros((total_T, sV), dtype=float)
        offset = 0
        for s in subj:
            T = s.sX.shape[0]
            Xcombined[offset:offset+T, :] = np.asarray(s.sX, dtype=float)
            offset += T

        # random initial seed 0..sV-1 (MATLAB used ceil(sV*rand))
        seed = rng.integers(low=0, high=sV)
        idx = furthest_sum(Xcombined, noc=noc, i=int(seed), exclude=None, treat_as_kernel=False)
        # Build C with ones at (idx[j], j)
        C = xp.zeros((sV, noc), dtype=float)
        C[idx, xp.arange(noc)] = 1.0
        muC = xp.array(1.0)
    else:
        C = xp.asarray(rng.random((sV, noc)), dtype=float)
        C /= (C.sum(axis=0, keepdims=True) + xp.finfo(float).eps)
        muC = xp.array(1.0)

    # ---------- Move subj arrays to GPU if requested; precompute SST ----------
    for s in subj:
        if runGPU:
            s.X  = cp.asarray(s.X, dtype=float)
            s.sX = cp.asarray(s.sX, dtype=float)
        s.T = s.sX.shape[0]
        s.SST = (s.X * s.X).sum()  # scalar (xp-type)

    SST = float(sum([float(s.SST.get()) if runGPU else float(s.SST) for s in subj]))

    # ---------- Initialize per-subject quantities ----------
    for s in subj:
        if voxelVariance:
            # Start at global average variance
            s.sigmaSq = xp.ones((V, 1), dtype=float) * (SST / (sum(T_list) * V))
        else:
            s.sigmaSq = xp.ones((V, 1), dtype=float)

        # Initialize S on the simplex (−log U then normalize)
        U = xp.asarray(rng.random((noc, V)), dtype=float)
        s.S = -xp.log(U + xp.finfo(float).tiny)
        s.S /= (s.S.sum(axis=0, keepdims=True) + xp.finfo(float).eps)

        s.muS = xp.ones((1,), dtype=float)  # will be expanded inside update

        # Sufficient statistics for the initial S
        s.sXC   = s.sX @ C                   # (T x sV) @ (sV x K) = (T x K)
        s.XCtX  = s.sXC.T @ s.X              # (K x T) @ (T x V) = (K x V)
        s.CtXtXC= s.sXC.T @ s.sXC            # (K x T) @ (T x K) = (K x K)
        s.SSt   = s.S @ s.S.T                # (K x K)
        s.XSt   = s.X @ s.S.T                # (T x V) @ (V x K) = (T x K)

        # Initialize S with a (potentially) longer line search (individual stepsize)
        S_np, muS_np, SSt_np = supdate_indi_step(
            _to_numpy(s.S, runGPU),
            _to_numpy(s.XCtX, runGPU),
            _to_numpy(s.CtXtXC, runGPU),
            np.ones(s.S.shape[1], dtype=float),   # per-column μ
            int(s.T),
            int(initial_S_steps),
            _to_numpy(s.sigmaSq.squeeze(), runGPU),
        )
        # Put back
        s.S   = xp.asarray(S_np)
        s.muS = xp.ones((s.S.shape[1],), dtype=float)  # reset to ones after init
        s.SSt = xp.asarray(SSt_np)

    # ---------- Initial NLL ----------
    NLL = 0.0
    SST_sigmaSq = 0.0
    for s in subj:
        s.XSt = s.X @ (s.S / s.sigmaSq.T).T           # (T x V) @ (V x K) = (T x K)
        s.SSt = s.S @ (s.S / s.sigmaSq.T).T           # (K x V) @ (V x K) = (K x K)
        s.SST_sigmaSq = (s.X * (s.X / s.sigmaSq.T)).sum()
        s.NLL = 0.5 * s.SST_sigmaSq \
                - (s.sXC * s.XSt).sum() \
                + 0.5 * (s.CtXtXC * s.SSt).sum() \
                + (s.T / 2.0) * (V * np.log(2.0 * np.pi) + xp.log(s.sigmaSq).sum())
        NLL += float(s.NLL.get()) if runGPU else float(s.NLL)
        SST_sigmaSq += float(s.SST_sigmaSq.get()) if runGPU else float(s.SST_sigmaSq)

    t_start = time.perf_counter()
    cost_fun = np.zeros((maxiter,), dtype=float)

    # Threshold for sigmaSq (numerical stability)
    noise_threshold_opt = mgetopt(opts, 'noise_threshold', None)
    if noise_threshold_opt is None:
        var_threshold = (SST / (sum(T_list) * V)) * 1e-3
    else:
        var_threshold = float(noise_threshold_opt)

    # ---------- Main loop ----------
    iter_ = 0
    dNLL = np.inf
    while ((abs(dNLL) >= conv_crit * abs(NLL)) or (fix_var_iter >= iter_)) and (iter_ < maxiter):
        iter_ += 1
        NLL_old = NLL
        cost_fun[iter_ - 1] = NLL

        # ---- C update (numCstep inner line searches) ----
        C, muC, NLL = _Cupdate_multi_subjects(
            subj, C, muC, NLL, numCstep, runGPU
        )

        # Update terms affected by new C
        for s in subj:
            s.sXC   = s.sX @ C
            s.XCtX  = s.sXC.T @ s.X
            s.CtXtXC= s.sXC.T @ s.sXC
            s.NLL   = 0.5 * s.SST_sigmaSq \
                      - (s.XCtX * (s.S / s.sigmaSq.T)).sum() \
                      + 0.5 * (s.CtXtXC * s.SSt).sum() \
                      + (s.T / 2.0) * (V * np.log(2.0 * np.pi) + xp.log(s.sigmaSq).sum())

        # ---- Update S (and sigmaSq if heteroscedastic & past warmup) ----
        NLL = 0.0
        SST_sigmaSq = 0.0
        for s in subj:
            S_np, muS_np, SSt_np = supdate_indi_step(
                _to_numpy(s.S, runGPU),
                _to_numpy(s.XCtX, runGPU),
                _to_numpy(s.CtXtXC, runGPU),
                _to_numpy(s.muS, runGPU),
                int(s.T),
                int(numSstep),
                _to_numpy(s.sigmaSq.squeeze(), runGPU),
            )
            s.S   = xp.asarray(S_np)
            s.muS = xp.asarray(muS_np)  # updated per-column steps
            s.SSt = xp.asarray(SSt_np)

            # Update voxel-specific variance
            if voxelVariance and (iter_ > fix_var_iter):
                resid = s.X - (s.sXC @ s.S)           # (T x V)
                s.sigmaSq = (resid * resid).sum(axis=0, keepdims=True).T / float(s.T)
                # Threshold
                s.sigmaSq = xp.maximum(s.sigmaSq, var_threshold)

                # Update sufficient stats
                s.XSt = s.X @ (s.S / s.sigmaSq.T).T
                s.SSt = s.S @ (s.S / s.sigmaSq.T).T

                # Update SST_sigmaSq
                s.SST_sigmaSq = (s.X * (s.X / s.sigmaSq.T)).sum()
            else:
                s.XSt = s.X @ (s.S / s.sigmaSq.T).T

            # Update NLL terms
            s.NLL = 0.5 * s.SST_sigmaSq \
                    - (s.sXC * s.XSt).sum() \
                    + 0.5 * (s.CtXtXC * s.SSt).sum() \
                    + (s.T / 2.0) * (V * np.log(2.0 * np.pi) + xp.log(s.sigmaSq).sum())

            NLL += float(s.NLL.get()) if runGPU else float(s.NLL)
            SST_sigmaSq += float(s.SST_sigmaSq.get()) if runGPU else float(s.SST_sigmaSq)

        dNLL = NLL_old - NLL

        # Optional: you can print progress every 5 iters
        # if (iter_ % 5) == 0:
        #     muS_meds = np.min([_to_numpy(s.muS, runGPU).median() for s in subj])
        #     sig_min  = np.min([_to_numpy(s.sigmaSq, runGPU).min() for s in subj])
        #     print(f"{iter_:12d} | {NLL:12.4e} | {(dNLL/abs(NLL)):12.4e} | {float(muC.get()) if runGPU else float(muC):12.4e} | {muS_meds:16.4e} | {sig_min:19.4e}")

        # Warn if NLL increased substantially
        if (dNLL / abs(NLL) < 0) and (abs(dNLL / NLL) > conv_crit):
            # print(f"Warning: NLL increased by {dNLL/abs(NLL):.4e}")
            pass

    # ---------- Wrap-up ----------
    elapsed = time.perf_counter() - t_start

    # SSE per subject (on CPU)
    SSE = []
    for s in subj:
        sXC_S = s.sXC @ s.S
        if runGPU:
            sse = float(cp.linalg.norm(s.X - sXC_S, ord='fro') ** 2)
        else:
            sse = float(np.linalg.norm(s.X - sXC_S, ord='fro') ** 2)
        SSE.append(sse)

    varexpl = (SST - sum(SSE)) / SST

    # Optional sorting of components
    ind = np.arange(noc)
    if sort_crit.lower() == 'corr' and (sum(T_list) == max(T_list) * B):
        # Compute mean inter-subject correlation per archetype (on CPU)
        arch = np.zeros((T_list[0], B))
        mean_corr = np.zeros((noc,))
        for j in range(noc):
            for bi, s in enumerate(subj):
                arch[:, bi] = _to_numpy(s.sXC[:, j], runGPU)
            Ccorr = np.corrcoef(arch, rowvar=False)
            # upper triangle mean
            iu = np.triu_indices(B, k=1)
            vals = Ccorr[iu]
            mean_corr[j] = float(vals.mean())
        ind = np.argsort(mean_corr)[::-1]
    elif sort_crit.lower() != 'energy':
        # fall back to energy with a warning (quiet by default)
        pass

    if not np.array_equal(ind, np.arange(noc)):
        # Reorder C and each subject's S and sXC
        C = C[:, ind]
        for s in subj:
            s.S   = s.S[ind, :]
            s.sXC = s.sXC[:, ind]

    # Gather to CPU results
    C_np = _to_numpy(C, runGPU)
    results_subj: List[Dict[str, Any]] = []
    for bi, s in enumerate(subj):
        out = {
            "S":            _to_numpy(s.S, runGPU),
            "sXC":          _to_numpy(s.sXC, runGPU),
            "sigmaSq":      _to_numpy(s.sigmaSq, runGPU).reshape(-1, 1),
            "NLL":          float(_to_numpy(s.NLL, runGPU)),
            "SSE":          float(SSE[bi]),
            "SST":          float(_to_numpy(s.SST, runGPU)),
            "SST_sigmaSq":  float(_to_numpy(s.SST_sigmaSq, runGPU)),
        }
        results_subj.append(out)

    return results_subj, C_np, cost_fun[:iter_], float(varexpl), float(elapsed)


# ------------------------- helpers -------------------------

def _to_numpy(a, runGPU: bool):
    """Bring xp array or scalar to numpy / Python types."""
    if runGPU and hasattr(a, "get"):
        try:
            return a.get()
        except Exception:
            # Scalars
            return np.array(a)
    return np.asarray(a)


def _Cupdate_multi_subjects(
    subj: List[Any],
    C,
    muC,
    NLL: float,
    niter: int,
    runGPU: bool,
):
    """
    Inner function: update C with niter line-search steps (matches MATLAB CupdateMultiSubjects).
    """
    xp = cp if runGPU else np
    sV, noc = C.shape
    V = subj[0].X.shape[1]

    # Constant terms: XtXSt = sX' * XSt
    XtXSt_list = []
    for s in subj:
        XtXSt_list.append(s.sX.T @ s.XSt)

    total_T = sum([int(s.T) for s in subj])

    for _ in range(niter):
        NLL_old = NLL

        # Gradient
        g = xp.zeros((sV, noc), dtype=float)
        for s, XtXSt in zip(subj, XtXSt_list):
            g += s.sX.T @ (s.sXC @ s.SSt) - XtXSt
        g /= (total_T * sV)

        # Project onto tangent space of column-simplex: g <- g - (sum(g*C) per column)
        col_dots = (g * C).sum(axis=0, keepdims=True)
        g = g - col_dots

        stop = False
        Cold = C.copy()
        while not stop:
            # Step and project back to simplex
            C = Cold - muC * g
            C = xp.maximum(C, 0.0)
            C /= (C.sum(axis=0, keepdims=True) + xp.finfo(float).eps)

            # Recompute NLL with new C
            NLL_tmp = 0.0
            for s in subj:
                s.sXC    = s.sX @ C
                s.CtXtXC = s.sXC.T @ s.sXC
                term = 0.5 * s.SST_sigmaSq \
                       - (s.sXC * s.XSt).sum() \
                       + 0.5 * (s.CtXtXC * s.SSt).sum() \
                       + (s.T / 2.0) * (V * np.log(2.0 * np.pi) + (cp if runGPU else np).log(s.sigmaSq).sum()
                                        )
                NLL_tmp += float(term.get()) if runGPU else float(term)

            if NLL_tmp <= NLL_old * (1.0 + 1e-9):
                muC = muC * 1.2
                NLL = NLL_tmp
                stop = True
            else:
                muC = muC / 2.0

    return C, muC, NLL