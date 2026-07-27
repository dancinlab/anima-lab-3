"""phi_rs.py — pure-Python drop-in replacing the removed Rust `phi_rs` extension.

@canonical-ok repo folder 'anima-lab-3' is the fixed project path, not a versioned copy.

'rust 제거': the phi-rs/ crate (PyO3/Rust, 625x speedup) was deleted; Phi is now computed by
phi_py.py (numpy), the SAME mutual-information IIT algorithm — slower, but zero build/toolchain.
This shim keeps every `import phi_rs` call site working (compute_phi is the hot path, ~20 uses),
so nothing that used the Rust extension breaks.
"""
from phi_py import compute_phi  # (states, n_bins=16, prev=None, curr=None, tensions=None) -> (phi, components)

__all__ = ["compute_phi", "search_combinations"]


def search_combinations(*args, **kwargs):
    """Rust-only mechanism screener; removed with the phi-rs crate.

    Was a native brute-force search; no pure-Python equivalent is shipped. Callers should
    measure Phi directly with compute_phi / phi_py instead.
    """
    raise NotImplementedError(
        "phi_rs.search_combinations was Rust-only and removed with the phi-rs crate; "
        "use phi_py.compute_phi for Phi measurement."
    )
