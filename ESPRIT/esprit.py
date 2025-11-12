import numpy as np
import scipy.signal
import scipy.linalg
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.cluster.vq import kmeans, vq  # Replaced sklearn with scipy for clustering
import json

# Small wrappers
def my_absd(x):
    return np.abs(x)

def my_fmin(a, b):
    return min(a, b)

def my_fmax(a, b):
    return max(a, b)

def my_isfinite(x):
    return np.isfinite(x)

def my_hypot(a, b):
    return np.hypot(a, b)

def frob_norm(A):
    return np.linalg.norm(A, 'fro')

# Eigenvalues for real-non symmetric matrix
def eigvals_real(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evals = np.linalg.eig(A)[0]
    lam_re = np.real(evals)
    lam_im = np.imag(evals)
    return lam_re, lam_im

# Build Hankel matrix
def build_hankel(x: np.ndarray, N: int, L: int) -> np.ndarray:
    C = N - L + 1
    if L < 2 or L >= N:
        raise ValueError("Invalid L")
    H = np.zeros((L, C))
    for i in range(L):
        H[i, :] = x[i:i + C]
    return H

# pinv KxK
def pinv_kxk(A: np.ndarray, tol: float) -> np.ndarray:
    return np.linalg.pinv(A, rcond=tol)

# Build U1 U2 from Us (L-major blocks)
def build_U1U2_Lmajor(Us: np.ndarray, rows: int, L: int, B: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    m = (L - 1) * B
    if rows != L * B:
        raise ValueError("Invalid rows")
    U1 = np.zeros((m, K))
    U2 = np.zeros((m, K))
    for b in range(B):
        base = b * L
        for ell in range(L - 1):
            rdst = b * (L - 1) + ell
            r1 = base + ell
            r2 = base + ell + 1
            U1[rdst, :] = Us[r1, :]
            U2[rdst, :] = Us[r2, :]
    return U1, U2

# Build V1 V2 from Vs
def build_V1V2(Vs: np.ndarray, C: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    m = C - 1
    V1 = Vs[0:m, :]
    V2 = Vs[1:C, :]
    return V1, V2

# TLS-ESPRIT from Us
def TLS_ESPRIT_FromUs(Us: np.ndarray, rows: int, L: int, B: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    TOL_PINV = 1e-9
    m = (L - 1) * B
    n2 = 2 * K
    sdim = min(m, n2)
    U1, U2 = build_U1U2_Lmajor(Us, rows, L, B, K)
    Z = np.zeros((m, n2))
    Z[:, 0:K] = U1
    Z[:, K:n2] = U2
    U_full, S, Vh = np.linalg.svd(Z, full_matrices=True)
    V = Vh.T
    smax = S[0] if len(S) > 0 else 0.0
    smin = S[sdim - 1] if sdim > 0 else 0.0
    cond = smax / smin if smin > 0 else np.inf
    print(f"[TLS-U] m={m} K={K} |Z| Smax={smax:.3e} Smin={smin:.3e} cond={cond:.3e}")
    Vn = V[:, -K:]
    X = Vn[0:K, :]
    Y = Vn[K:n2, :]
    Yp = pinv_kxk(Y, TOL_PINV)
    Phi = - (X @ Yp)  # Fixed sign
    R = X + (Phi @ Y)  # Fixed for sign
    nr = frob_norm(R)
    ny = frob_norm(Y)
    res = nr / ny if ny > 0 else 0.0
    print(f"[TLS-U] residual = {res:.3e}")
    lam_re, lam_im = eigvals_real(Phi)
    r_abs = np.hypot(lam_re, lam_im)
    mn = np.min(r_abs) if len(r_abs) > 0 else 0.0
    mx = np.max(r_abs) if len(r_abs) > 0 else 0.0
    print(f"[TLS-U] |lambda| range: [{mn:.6f} .. {mx:.6f}]")
    for t in range(K):
        print(f"[TLS-U lambda {t}] {lam_re[t]:.6f} + {lam_im[t]:.6f}i")
    return lam_re, lam_im

# TLS-ESPRIT from Vs
def TLS_ESPRIT_FromVs(Vs: np.ndarray, C: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    TOL_PINV = 1e-9
    m = C - 1
    n2 = 2 * K
    sdim = min(m, n2)
    V1, V2 = build_V1V2(Vs, C, K)
    Z = np.zeros((m, n2))
    Z[:, 0:K] = V1
    Z[:, K:n2] = V2
    U_full, S, Vh = np.linalg.svd(Z, full_matrices=True)
    V = Vh.T
    smax = S[0] if len(S) > 0 else 0.0
    smin = S[sdim - 1] if sdim > 0 else 0.0
    cond = smax / smin if smin > 0 else np.inf
    print(f"[TLS-V] m={m} K={K} |Z| Smax={smax:.3e} Smin={smin:.3e} cond={cond:.3e}")
    Vn = V[:, -K:]
    X = Vn[0:K, :]
    Y = Vn[K:n2, :]
    Yp = pinv_kxk(Y, TOL_PINV)
    Phi = - (X @ Yp)  # Fixed sign
    R = X + (Phi @ Y)  # Fixed
    nr = frob_norm(R)
    ny = frob_norm(Y)
    res = nr / ny if ny > 0 else 0.0
    print(f"[TLS-V] residual = {res:.3e}")
    lam_re, lam_im = eigvals_real(Phi)
    r_abs = np.hypot(lam_re, lam_im)
    mn = np.min(r_abs) if len(r_abs) > 0 else 0.0
    mx = np.max(r_abs) if len(r_abs) > 0 else 0.0
    print(f"[TLS-V] |lambda| range: [{mn:.6f} .. {mx:.6f}]")
    for t in range(K):
        print(f"[TLS-V lambda {t}] {lam_re[t]:.6f} + {lam_im[t]:.6f}i")
    return lam_re, lam_im

# LS-ESPRIT from Vs
def LS_ESPRIT_FromVs(Vs: np.ndarray, C: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    TOL = 1e-8
    V1, V2 = build_V1V2(Vs, C, K)
    m = C - 1
    G = V1.T @ V1
    Gp = pinv_kxk(G, TOL)
    Phi = Gp @ (V1.T @ V2)
    lam_re, lam_im = eigvals_real(Phi)
    r_abs = np.hypot(lam_re, lam_im)
    mn = np.min(r_abs) if len(r_abs) > 0 else 0.0
    mx = np.max(r_abs) if len(r_abs) > 0 else 0.0
    print(f"[LS-V] |lambda| range: [{mn:.6f} .. {mx:.6f}]")
    for t in range(K):
        print(f"[LS-V lambda {t}] {lam_re[t]:.6f} + {lam_im[t]:.6f}i")
    return lam_re, lam_im

# Lambdas to f, Q, zeta
def LambdasToFZQ(lam_re: np.ndarray, lam_im: np.ndarray, K: int, dt: float, max_modes: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    RMIN = 0.50
    RMAX = 1.30
    IM_TOL = 1e-6
    BIGQ = 1e6
    fs = 1.0 / dt
    nyq = 0.5 * fs
    fmin = 30.0
    fmax = 0.95 * nyq
    if max_modes <= 0:
        max_modes = K // 2
    used = np.zeros(K, dtype=bool)
    F_list = []
    Q_list = []
    Z_list = []
    for i in range(K):
        if used[i]:
            continue
        ai = lam_re[i]
        bi = lam_im[i]
        ri = my_hypot(ai, bi)
        if my_absd(bi) < IM_TOL:
            continue
        if not (RMIN <= ri <= RMAX):
            continue
        best = np.inf
        jbest = -1
        for j in range(i + 1, K):
            if used[j]:
                continue
            aj = lam_re[j]
            bj = lam_im[j]
            rj = my_hypot(aj, bj)
            if my_absd(bj + bi) > 1e-5 * (my_absd(bj) + my_absd(bi) + 1.0):
                continue
            if not (RMIN <= rj <= RMAX):
                continue
            err = my_absd(aj - ai) + 1.0 * my_absd(bj + bi) + my_absd(rj - ri)
            if err < best:
                best = err
                jbest = j
        if jbest == -1:
            continue
        used[i] = True
        used[jbest] = True
        a = 0.5 * (ai + lam_re[jbest])
        b = 0.5 * (bi - lam_im[jbest])
        r = my_hypot(a, b)
        th = np.arctan2(b, a)
        s_re = np.log(max(r, 1e-300)) / dt
        s_im = th / dt
        f = my_absd(s_im) / (2.0 * np.pi)
        w = 2.0 * np.pi * f
        alpha = -s_re
        zeta = 0.0
        Qv = BIGQ
        if f > 0.0 and my_isfinite(alpha):
            den = np.sqrt(alpha**2 + w**2)
            if den > 0.0:
                zeta = alpha / den
            zeta = max(0.0, min(zeta, 0.5))
            Qv = 1.0 / (2.0 * zeta) if zeta > 0.0 else BIGQ
        print(f"[MODE raw] f={f:.2f} Hz Q={Qv:.2f} zeta={zeta:.4f}")
        if f <= fmin or f >= fmax:
            continue
        F_list.append(f)
        Q_list.append(Qv)
        Z_list.append(zeta)
        if len(F_list) >= max_modes:
            break
    F = np.array(F_list)
    Q = np.array(Q_list)
    Z = np.array(Z_list)
    if len(F) > 1:
        sort_idx = np.argsort(F)
        F = F[sort_idx]
        Q = Q[sort_idx]
        Z = Z[sort_idx]
    M = len(F)
    if M > 0:
        print(f"[MAP] dt={dt:.6g} s, modes={M}; f[0]={F[0]:.2f}..f[{M-1}]={F[-1]:.2f} Hz")
    else:
        print(f"[MAP] No valid complex-conjugate modes found (K={K})")
    return F, Q, Z

# Synthetic signal
def synth_signal(x: np.ndarray, N: int, fs: float, f_true: np.ndarray, Q_true: np.ndarray, A_true: np.ndarray, ph_true: np.ndarray, Ksig: int):
    for k in range(Ksig):
        f = f_true[k]
        Q = Q_true[k]
        A = A_true[k]
        ph = ph_true[k]
        w = 2.0 * np.pi * f
        alpha = w / (2.0 * Q)
        t = np.arange(N) / fs
        x += A * np.exp(-alpha * t) * np.cos(w * t + ph)

# Self-test function
def self_test():
    fs = 787.815125
    dt = 1.0 / fs
    N = 260
    L = 140
    C = N - L + 1
    Ksub = 12
    Jwant = 6
    f_true = np.array([120, 145, 168, 185, 210, 235])
    Q_true = np.array([12, 20, 15, 8, 25, 18])
    print("Ground truth (f,Q):")
    for i in range(6):
        print(f" #{i} f={f_true[i]:.2f} Hz, Q={Q_true[i]:.2f}")
    x = np.zeros(N)
    A_true = np.array([1.0, 0.9, 0.8, 0.7, 0.65, 0.6])
    ph_true = np.array([0.2, -0.6, 0.9, -1.1, 0.7, -0.3])
    synth_signal(x, N, fs, f_true, Q_true, A_true, ph_true, 6)
    H = build_hankel(x, N, L)
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh.T
    print(f"[SELFTEST] Top singular values: S1={S[0]:.3e} S2={S[1]:.3e} S3={S[2]:.3e}")
    Us = U[:, :Ksub]
    Vs = V[:, :Ksub]
    lam_re, lam_im = TLS_ESPRIT_FromVs(Vs, C, Ksub)
    F, Q, Z = LambdasToFZQ(lam_re, lam_im, Ksub, dt, Jwant)
    if len(F) >= 3:
        for k in range(len(F)):
            print(f"[MODE V {k}] f={F[k]:.2f} Hz Q={Q[k]:.2f} zeta={Z[k]:.4f}")
    else:
        lam_re, lam_im = TLS_ESPRIT_FromUs(Us, L, L, 1, Ksub)
        F, Q, Z = LambdasToFZQ(lam_re, lam_im, Ksub, dt, Jwant)
        if len(F) >= 3:
            for k in range(len(F)):
                print(f"[MODE U {k}] f={F[k]:.2f} Hz Q={Q[k]:.2f} zeta={Z[k]:.4f}")
        else:
            lam_re, lam_im = LS_ESPRIT_FromVs(Vs, C, Ksub)
            F, Q, Z = LambdasToFZQ(lam_re, lam_im, Ksub, dt, Jwant)
            for k in range(len(F)):
                print(f"[MODE LS {k}] f={F[k]:.2f} Hz Q={Q[k]:.2f} zeta={Z[k]:.4f}")
    print("[SELFTEST] Done.")
    # Reconstruct and plot
    num_modes = len(F)
    if num_modes > 0:
        t = np.arange(N) / fs
        basis = np.zeros((N, 2 * num_modes))
        for k in range(num_modes):
            f = F[k]
            Qv = Q[k]
            w = 2 * np.pi * f
            alpha = w / (2 * Qv)
            exp_a = np.exp(-alpha * t)
            basis[:, 2 * k] = exp_a * np.cos(w * t)
            basis[:, 2 * k + 1] = exp_a * np.sin(w * t)
        coeff = np.linalg.lstsq(basis, x, rcond=None)[0]
        x_rec = basis @ coeff
        plt.figure()
        plt.plot(t, x, label='Original')
        plt.plot(t, x_rec, label='Reconstructed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Self-test Reconstruction')
        plt.show()

# Cube loading and processing
class BandPreset:
    def __init__(self, low_freq: float, high_freq: float, decimate_factor: int, N_band: int, filter_order: int, exp_alpha: float):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.decimate_factor = decimate_factor
        self.N_band = N_band
        self.filter_order = filter_order
        self.exp_alpha = exp_alpha

band_presets = [
    BandPreset(40.0, 500.0, 4, 8192, 6, 0.01),
    BandPreset(500.0, 1000.0, 2, 8192, 6, 0.005),
    BandPreset(1000.0, 2000.0, 1, 8192, 8, 0.002),
    BandPreset(2000.0, 4000.0, 1, 8192, 8, 0.001),
]

def read_index(index_path: str) -> Tuple[int, int, int, int, float, List[str]]:
    with open(index_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].replace('#', '').strip()
    parts = header.split()
    R = int(parts[0].split('=')[1])
    M_raw = int(parts[1].split('=')[1])
    M_out = int(parts[2].split('=')[1])
    N_use = int(parts[3].split('=')[1])
    fs = float(parts[4].split('=')[1])
    fs = abs(fs)  # Ensure positive fs
    names = [line.strip() for line in lines[1:] if line.strip() and not line.startswith('#')]
    if len(names) != R:
        raise ValueError("Names count mismatch")
    return R, M_raw, M_out, N_use, fs, names

def load_cube(cube_path: str, R: int, M_out: int, N_use: int) -> np.ndarray:
    flat = np.fromfile(cube_path, dtype=np.float64)
    if len(flat) != R * M_out * N_use:
        raise ValueError("Size mismatch")
    return flat.reshape((R, M_out, N_use))

# Process cube for selected band
def process_cube(y_cube: np.ndarray, R: int, M_out: int, N_use: int, fs: float, current_preset: BandPreset, skip_m: int = 2) -> Tuple[np.ndarray, float]:
    m_map = [m for m in range(M_out) if m != skip_m]
    M_eff = len(m_map)
    processed_cube = np.zeros((R, M_eff, current_preset.N_band))
    fs_band = fs / current_preset.decimate_factor
    for r in range(R):
        for me in range(M_eff):
            m = m_map[me]
            y_orig = y_cube[r, m, :N_use]
            # Bandpass
            sos = scipy.signal.butter(current_preset.filter_order, [current_preset.low_freq / (fs / 2), current_preset.high_freq / (fs / 2)], btype='bandpass', analog=False, output='sos')
            y_band = scipy.signal.sosfilt(sos, y_orig)
            # Exp window
            t = np.arange(N_use) / fs
            y_band *= np.exp(-current_preset.exp_alpha * t)
            # Decimate
            if current_preset.decimate_factor > 1:
                y_band = scipy.signal.decimate(y_band, current_preset.decimate_factor, ftype='iir')
            # Truncate
            N_proc = len(y_band)
            N_band = min(N_proc, current_preset.N_band)
            processed_cube[r, me, :N_band] = y_band[:N_band]
    return processed_cube, fs_band

# Parse selected r string, e.g., "1,3-5,7"
def parse_selected_r(selected_r_str: str, R_total: int) -> List[int]:
    if not selected_r_str:
        return list(range(R_total))
    selected = []
    parts = selected_r_str.split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))
    return sorted(set(selected))  # Unique and sorted

# Example usage for data
def run_on_data(index_path: str, cube_path: str, band_index: int = 0, L_fraction: float = 0.5, K: int = 12, r_index: int = 0):
    R, M_raw, M_out, N_use, fs, names = read_index(index_path)
    y_cube = load_cube(cube_path, R, M_out, N_use)
    current_preset = band_presets[band_index]
    processed_cube, fs_band = process_cube(y_cube, R, M_out, N_use, fs, current_preset)
    N_band = processed_cube.shape[2]
    dt = 1.0 / fs_band
    L = int(N_band * L_fraction)
    C = N_band - L + 1
    M_eff = processed_cube.shape[1]
    # For single r, multi-channel
    big_H = np.vstack([build_hankel(processed_cube[r_index, me, :N_band], N_band, L) for me in range(M_eff)])
    U, S, Vh = np.linalg.svd(big_H, full_matrices=False)
    V = Vh.T
    Us = U[:, :K]
    lam_re, lam_im = TLS_ESPRIT_FromUs(Us, M_eff * L, L, M_eff, K)
    F, Q, Z = LambdasToFZQ(lam_re, lam_im, K, dt)
    # Plot example response for r, me=0
    me = 0
    t = np.arange(N_band) / fs_band
    plt.figure()
    plt.plot(t, processed_cube[r_index, me, :N_band])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Processed Response r={r_index} me={me}')
    plt.show()

# Updated stabilization_diagram with mode shapes
def stabilization_diagram_with_shapes(index_path: str, cube_path: str, band_index: int = 0, L_fraction: float = 0.5, K: int = 30, selected_r: str = "", output_file: str = "results.json"):
    R_total, M_raw, M_out, N_use, fs, names_total = read_index(index_path)
    selected_r_indices = parse_selected_r(selected_r, R_total)
    R = len(selected_r_indices)
    print(f"Processing {R} selected excitation points: {selected_r_indices}")
    y_cube_total = load_cube(cube_path, R_total, M_out, N_use)
    y_cube = y_cube_total[selected_r_indices, :, :]
    names = [names_total[i] for i in selected_r_indices]
    current_preset = band_presets[band_index]
    processed_cube, fs_band = process_cube(y_cube, R, M_out, N_use, fs, current_preset)
    N_band = processed_cube.shape[2]
    dt = 1.0 / fs_band
    L = int(N_band * L_fraction)
    C = N_band - L + 1
    M_eff = processed_cube.shape[1]
    all_f = []
    all_z = []
    all_r = []
    all_lam_re = []  # To store lambdas per r
    all_lam_im = []
    for ri, r in enumerate(selected_r_indices):
        try:
            print(f"[PROCESS] Processing excitation point r={r} (index {ri})")
            big_H = np.vstack([build_hankel(processed_cube[ri, me, :N_band], N_band, L) for me in range(M_eff)])
            U, S, Vh = np.linalg.svd(big_H, full_matrices=False)
            V = Vh.T
            Us = U[:, :K]
            lam_re, lam_im = TLS_ESPRIT_FromUs(Us, M_eff * L, L, M_eff, K)
            F, Q, Z = LambdasToFZQ(lam_re, lam_im, K, dt)
            all_f.extend(F)
            all_z.extend(Z)
            all_r.extend([r] * len(F))
            all_lam_re.append(lam_re)
            all_lam_im.append(lam_im)
        except Exception as e:
            print(f"Error for r={r}: {e}")
    # Plot stabilization
    plt.figure()
    plt.scatter(all_r, all_f, s=5)
    plt.xlabel('Excitation Point (r)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Stabilization Diagram')
    plt.show()
    # Cluster to find common modes (estimate num_clusters by density)
    if len(all_f) == 0:
        print("No modes found")
        return
    f_array = np.array(all_f).reshape(-1, 1)
    num_clusters = min(20, len(all_f) // 2)  # Adjust as needed
    centroids, _ = kmeans(f_array, num_clusters)
    labels, _ = vq(f_array, centroids)
    unique_labels = np.unique(labels)
    common_f = []
    common_z = []
    for lab in unique_labels:
        idx = labels == lab
        if np.sum(idx) > 0:
            fs_cluster = np.array(all_f)[idx]
            zs_cluster = np.array(all_z)[idx]
            mean_f = np.mean(fs_cluster)
            mean_z = np.mean(zs_cluster)
            if not np.isnan(mean_f) and not np.isnan(mean_z):
                common_f.append(mean_f)
                common_z.append(mean_z)
    # Sort by frequency
    if len(common_f) > 0:
        sort_idx = np.argsort(common_f)
        common_f = [common_f[i] for i in sort_idx]
        common_z = [common_z[i] for i in sort_idx]
    print(f"Common modes: {len(common_f)}")
    for i in range(len(common_f)):
        print(f"Mode {i}: f={common_f[i]:.2f} Hz, zeta={common_z[i]:.4f}")
    # Fit amplitudes for shapes
    t = np.arange(N_band) * dt
    mode_shapes_along_r = []  # List of complex A_r for each mode
    amplitudes_in_m = []  # List of avg A_m per mode (across r)
    participation = []  # |A_r| per mode along r
    signed_shapes = []  # Real signed shapes
    for k in range(len(common_f)):
        f_k = common_f[k]
        zeta_k = common_z[k]
        if np.isnan(zeta_k) or zeta_k < 0 or zeta_k >= 1:
            continue  # Skip invalid zeta
        w_k = 2 * np.pi * f_k
        if zeta_k < 1:
            alpha_k = zeta_k * w_k / np.sqrt(1 - zeta_k**2)
        else:
            alpha_k = zeta_k * w_k
        exp_a = np.exp(-alpha_k * t[:N_band])
        cos_term = exp_a * np.cos(w_k * t[:N_band])
        sin_term = exp_a * np.sin(w_k * t[:N_band])
        A_r_complex = np.zeros(R, dtype=complex)
        A_m_per_r = np.zeros((R, M_eff), dtype=complex)  # For amplitudes in m
        for ri in range(R):
            A_me = np.zeros(M_eff, dtype=complex)
            for me in range(M_eff):
                y = processed_cube[ri, me, :N_band]
                basis = np.column_stack([cos_term, sin_term])
                try:
                    coeff = np.linalg.lstsq(basis, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    coeff = np.zeros(2)
                a, b = coeff if len(coeff) == 2 else (0.0, 0.0)
                A_me[me] = a + 1j * b  # Complex amplitude
            A_r_complex[ri] = np.mean(A_me)  # Avg over m for shape along r
            A_m_per_r[ri, :] = A_me
        mode_shapes_along_r.append(A_r_complex)
        amplitudes_in_m.append(np.mean(A_m_per_r, axis=0))  # Avg over r for each m
        participation.append(np.abs(A_r_complex))
        # Compute signed real shape: multiply by phase to make real
        if np.any(A_r_complex != 0):
            ref_phase = np.angle(A_r_complex[np.argmax(np.abs(A_r_complex))])
            signed_shape = np.real(A_r_complex * np.exp(-1j * ref_phase))
            signed_shape /= np.max(np.abs(signed_shape)) if np.max(np.abs(signed_shape)) != 0 else 1  # Normalize
        else:
            signed_shape = np.zeros(R)
        signed_shapes.append(signed_shape)
    # Save results
    import os;
    print(os.getcwd())
    results = {
        'common_f': common_f,
        'common_z': common_z,
        'signed_shapes': [s.tolist() for s in signed_shapes],
        'participation': [p.tolist() for p in participation],
        'amplitudes_in_m': [[{"real": c.real, "imag": c.imag} for c in a] for a in amplitudes_in_m],
        'names': names,
        'selected_r_indices': selected_r_indices
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
    # Interactive plotting
    class ModeViewer:
        def __init__(self, common_f, signed_shapes, names, r_points):
            self.common_f = common_f
            self.signed_shapes = signed_shapes
            self.names = names
            self.r_points = r_points
            self.k = 0
            self.fig, self.ax = plt.subplots()
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.update_plot()

        def on_key(self, event):
            if event.key == 'right':
                self.k = (self.k + 1) % len(self.common_f)
            elif event.key == 'left':
                self.k = (self.k - 1) % len(self.common_f)
            self.update_plot()

        def update_plot(self):
            self.ax.clear()
            self.ax.plot(self.r_points, self.signed_shapes[self.k], label='Signed Amplitude')
            self.ax.set_xlabel('Excitation points (from index.txt)')
            self.ax.set_ylabel('Signed Amplitude')
            self.ax.set_title(f'Mode {self.k}: f={self.common_f[self.k]:.2f} Hz - Signed Shape along bridge')
            self.ax.set_xticks(self.r_points)
            self.ax.set_xticklabels(self.names, rotation=90)
            self.fig.canvas.draw()

    if len(common_f) > 0:
        r_points = np.arange(R)
        viewer = ModeViewer(common_f, signed_shapes, names, r_points)
        plt.show()
    # Participation and amplitudes
    for k in range(len(common_f)):
        print(f"Mode {k}: Participation factors along r: {participation[k]}")
        amp_m = amplitudes_in_m[k]
        print(f"Mode {k}: Amplitudes in receivers (complex with sign): {amp_m}")

# Run self-test
self_test()

# Example call (uncomment and set paths if needed, add selected_r='1,3-5' for example)
stabilization_diagram_with_shapes("D:\\NEUMANN\\Modes_measurements\\out\\index.txt", "D:\\NEUMANN\\Modes_measurements\\out\\y_cube.bin", band_index=3, K=40, selected_r="")