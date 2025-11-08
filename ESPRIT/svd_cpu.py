
"""
svd_cpu.py
Economy SVD (thin) on a 2D numpy array.
Usage:
    from svd_cpu import svd_cpu
    U, s, Vt, secs = svd_cpu(A)
"""
from __future__ import annotations
import numpy as np
import time

def svd_cpu(A: np.ndarray):
    """
    Compute thin SVD on CPU with NumPy (LAPACK).
    Returns: U, s, Vt, elapsed_seconds
    """
    t0 = time.perf_counter()
    # economy SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    secs = time.perf_counter() - t0
    return U, s, Vt, secs

if __name__ == "__main__":
    import sys, json
    path = sys.argv[1]
    A = np.load(path)
    U, s, Vt, secs = svd_cpu(A)
    print(json.dumps({
        "shape": list(A.shape),
        "U_shape": list(U.shape),
        "s_len": len(s),
        "Vt_shape": list(Vt.shape),
        "elapsed_sec": secs
    }, indent=2))
