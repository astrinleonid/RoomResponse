
"""
run_svd_profile.py  (capped)
Loads a saved Hankel matrix and profiles CPU and (optionally) GPU SVD.
To keep execution fast in limited environments, we cap the number of columns.
Saves timing JSON to /mnt/data/svd_timings.json
"""
import json, time, os, numpy as np

from svd_cpu import svd_cpu

CAP_COLS = int(os.environ.get("HANKEL_CAP_COLS", "4000"))

def try_gpu(A):
    try:
        from svd_gpu import svd_gpu
        U, s, Vt, secs, used = svd_gpu(A)
        if not used:
            return {"gpu_available": False}
        return {
            "gpu_available": True,
            "elapsed_sec": secs,
            "U_shape": list(U.shape),
            "s_len": len(s),
            "Vt_shape": list(Vt.shape),
        }
    except Exception as e:
        return {"gpu_available": False, "error": str(e)}

def main(hankel_path: str, out_json: str):
    A_full = np.load(hankel_path)
    K = A_full.shape[1]
    Kcap = min(CAP_COLS, K)
    A = A_full[:, :Kcap].copy()

    # CPU
    Uc, sc, Vtc, cpu_secs = svd_cpu(A)
    cpu_res = {
        "elapsed_sec": cpu_secs,
        "U_shape": list(Uc.shape),
        "s_len": len(sc),
        "Vt_shape": list(Vtc.shape),
        "cols_used": Kcap
    }
    # GPU (optional)
    gpu_res = try_gpu(A)
    gpu_res["cols_used"] = Kcap

    # Persist a couple of small artifacts
    import pathlib
    output_dir = pathlib.Path(hankel_path).parent
    np.save(output_dir / "Sigma.npy", sc)
    # Save first 8 singular values for quick glance
    first8 = sc[:8].tolist()

    result = {
        "matrix_shape_full": list(A_full.shape),
        "matrix_shape_used": list(A.shape),
        "cpu": cpu_res,
        "gpu": gpu_res,
        "sigma_first8": first8
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import sys
    import pathlib
    if len(sys.argv) < 2:
        print("Usage: python run_svd_profile.py <path_to_hankel.npy> [--full] [--cap N]")
        raise SystemExit(2)
    hankel_path = sys.argv[1]
    # Allow --full flag to override CAP_COLS
    if "--full" in sys.argv:
        CAP_COLS = 999999  # effectively unlimited
    # Allow --cap N to set specific column count
    if "--cap" in sys.argv:
        cap_idx = sys.argv.index("--cap")
        if cap_idx + 1 < len(sys.argv):
            CAP_COLS = int(sys.argv[cap_idx + 1])
    output_dir = pathlib.Path(hankel_path).parent
    out_json = str(output_dir / "svd_timings.json")
    main(hankel_path, out_json)
