
"""
svd_gpu.py
Economy SVD (thin) on a 2D CuPy array.
Usage:
    from svd_gpu import svd_gpu
    U, s, Vt, secs = svd_gpu(A_np)
"""
from __future__ import annotations
import time

def svd_gpu(A_host):
    """
    Try to compute thin SVD on GPU with CuPy.
    - A_host: numpy array (host). Will be copied to device.
    Returns: (U, s, Vt, secs, used_gpu) where arrays are on host (numpy).
             If CuPy or a CUDA device is unavailable, returns (None, None, None, None, False).
    """
    try:
        import cupy as cp
        # Ensure there is at least one device
        _ = cp.cuda.runtime.getDeviceCount()
    except Exception:
        return None, None, None, None, False

    start_total = time.perf_counter()

    # Transfer to device
    dA = cp.asarray(A_host)

    # synchronize before timing SVD
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    dU, ds, dVt = cp.linalg.svd(dA, full_matrices=False)
    cp.cuda.Stream.null.synchronize()
    secs = time.perf_counter() - t0

    # Bring results back to host (optional, but useful for parity)
    U = cp.asnumpy(dU)
    s = cp.asnumpy(ds)
    Vt = cp.asnumpy(dVt)

    total = time.perf_counter() - start_total

    return U, s, Vt, secs, True

if __name__ == "__main__":
    import sys, json, numpy as np
    path = sys.argv[1]
    A = np.load(path)
    U, s, Vt, secs, used_gpu = svd_gpu(A)
    if not used_gpu:
        print(json.dumps({"gpu_available": False}))
    else:
        print(json.dumps({
            "gpu_available": True,
            "shape": list(A.shape),
            "U_shape": list(U.shape),
            "s_len": len(s),
            "Vt_shape": list(Vt.shape),
            "elapsed_sec": secs
        }, indent=2))
