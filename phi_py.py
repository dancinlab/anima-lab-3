"""phi_py — pure-Python (numpy) port of phi_rs.compute_phi.

@canonical-ok repo folder 'anima-lab-3' is the fixed project path, not a versioned copy.

Faithful reimplementation of the Rust Φ(IIT) calculator (phi-rs/src/lib.rs) so
Φ can be measured WITHOUT building the Rust extension. Same algorithm:

    Φ = spatial_phi + temporal_phi * 0.5 + complexity * 0.1
      spatial_phi = (total_MI - min_partition_MI) / (n - 1)
      temporal_phi = mean_i MI(h_prev_i, h_curr_i)                (0 if no temporal states)
      complexity   = entropy(tensions)  OR  std(row_sums of MI)

MI between two cell state vectors is histogram-based: H(A)+H(B)-H(A,B), clamped >=0.

NOTE ON SPEED: the Rust version is ~625x faster. For N cells the pairwise MI matrix
is O(N^2 * dim). Callers doing this every training step should subsample cells
(see `compute_phi_subsampled`) — pure-Python over 256 cells per step is too slow.
"""
import numpy as np


def _bin_values(v, n_bins):
    v = np.asarray(v, dtype=np.float64)
    lo = v.min()
    rng = v.max() - lo
    if rng < 1e-12:
        return np.zeros(v.shape, dtype=np.int64)
    return np.clip(((v - lo) / rng * n_bins).astype(np.int64), 0, n_bins - 1)


def _entropy(counts, total):
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def _mi_paired(a, b, n_bins):
    n = len(a)
    if n == 0:
        return 0.0
    ba, bb = _bin_values(a, n_bins), _bin_values(b, n_bins)
    ca = np.bincount(ba, minlength=n_bins)
    cb = np.bincount(bb, minlength=n_bins)
    joint = np.bincount(ba * n_bins + bb, minlength=n_bins * n_bins)
    return max(0.0, _entropy(ca, n) + _entropy(cb, n) - _entropy(joint, n))


def _mi_matrix(S, n_bins):
    n = S.shape[0]
    mi = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = _mi_paired(S[i], S[j], n_bins)
            mi[i, j] = mi[j, i] = m
    return mi


def _min_partition(mi, n):
    if n <= 1:
        return 0.0
    if n == 2:
        return float(mi[0, 1])
    if n <= 20:  # exact: every non-trivial bipartition with cell 0 pinned to A
        best = np.inf
        for mask in range(1, 1 << (n - 1)):
            a, b = [0], []
            for bit in range(n - 1):
                (a if (mask >> bit) & 1 else b).append(bit + 1)
            if b:
                best = min(best, mi[np.ix_(a, b)].sum())
        return float(best)
    # greedy: sort cells by total MI, sweep the split, take the minimum cut
    order = np.argsort(mi.sum(axis=1))
    best = np.inf
    for split in range(1, n):
        a, b = order[:split], order[split:]
        best = min(best, mi[np.ix_(a, b)].sum())
    return float(best)


def _distribution_entropy(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return 0.0
    shifted = v - v.min()
    total = shifted.sum()
    if total < 1e-8:
        return 0.0
    p = shifted / total
    return float(max(0.0, -(p * np.log2(p + 1e-10)).sum()))


def compute_phi(states, n_bins=16, prev_states=None, curr_states=None, tensions=None):
    """Return (phi, components) — matches phi_rs.compute_phi output shape."""
    S = np.asarray(states, dtype=np.float64)
    n = S.shape[0]
    if n <= 1:
        return 0.0, {"spatial": 0.0, "temporal": 0.0, "complexity": 0.0}

    mi = _mi_matrix(S, n_bins)
    total = mi[np.triu_indices(n, 1)].sum()
    min_part = _min_partition(mi, n)
    spatial = max(0.0, total - min_part) / max(1.0, n - 1.0)

    temporal = 0.0
    if prev_states is not None and curr_states is not None:
        P, C = np.asarray(prev_states), np.asarray(curr_states)
        temporal = sum(_mi_paired(P[i], C[i], n_bins) for i in range(n)) / n

    if tensions is not None:
        complexity = _distribution_entropy(tensions)
    else:
        complexity = float(mi.sum(axis=1).std())

    phi = spatial + temporal * 0.5 + complexity * 0.1
    return float(phi), {"spatial": spatial, "temporal": temporal, "complexity": complexity}


def compute_phi_subsampled(states, n_bins=16, max_cells=32):
    """Fast path for per-step use: compute Φ on up to `max_cells` cells.

    Deterministic stride subsample (no RNG — safe for resume). Keeps the pure
    -Python cost bounded (~O(max_cells^2)) so it can run every training step.
    """
    S = np.asarray(states)
    n = S.shape[0]
    if n > max_cells:
        idx = np.linspace(0, n - 1, max_cells).astype(np.int64)
        S = S[idx]
    phi, _ = compute_phi(S, n_bins=n_bins)
    return phi
