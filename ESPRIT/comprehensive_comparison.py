"""
comprehensive_comparison.py
Robust comparison between esprit_core.py and esprit.py implementations.

Three levels of analysis:
1. Single-point analysis with side-by-side visualization
2. Single frequency range multi-point analysis with side-by-side visualization
3. Full dataset analysis with statistical comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import scipy.signal

# Import both implementations
import esprit_core

# Import esprit.py functions without running module-level code
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("esprit_ref", "ESPRIT/esprit.py")
esprit_ref = importlib.util.module_from_spec(spec)
sys.modules["esprit_ref"] = esprit_ref
# Don't execute the module yet - we'll import specific functions manually

# Define the functions we need from esprit.py directly
exec("""
import numpy as np
import scipy.signal
import scipy.linalg

def build_hankel(x: np.ndarray, N: int, L: int) -> np.ndarray:
    C = N - L + 1
    if L < 2 or L >= N:
        raise ValueError("Invalid L")
    H = np.zeros((L, C))
    for i in range(L):
        H[i, :] = x[i:i + C]
    return H

def frob_norm(A):
    return np.linalg.norm(A, 'fro')

def pinv_kxk(A: np.ndarray, tol: float) -> np.ndarray:
    return np.linalg.pinv(A, rcond=tol)

def build_U1U2_Lmajor(Us: np.ndarray, rows: int, L: int, B: int, K: int):
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

def eigvals_real(A: np.ndarray):
    evals = np.linalg.eig(A)[0]
    lam_re = np.real(evals)
    lam_im = np.imag(evals)
    return lam_re, lam_im

def TLS_ESPRIT_FromUs(Us: np.ndarray, rows: int, L: int, B: int, K: int):
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
    Vn = V[:, -K:]
    X = Vn[0:K, :]
    Y = Vn[K:n2, :]
    Yp = pinv_kxk(Y, TOL_PINV)
    Phi = - (X @ Yp)
    R = X + (Phi @ Y)
    nr = frob_norm(R)
    ny = frob_norm(Y)
    res = nr / ny if ny > 0 else 0.0
    lam_re, lam_im = eigvals_real(Phi)
    return lam_re, lam_im

def LambdasToFZQ(lam_re: np.ndarray, lam_im: np.ndarray, K: int, dt: float, max_modes: int = 0):
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
        ri = np.hypot(ai, bi)
        if np.abs(bi) < IM_TOL:
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
            rj = np.hypot(aj, bj)
            if np.abs(bj + bi) > 1e-5 * (np.abs(bj) + np.abs(bi) + 1.0):
                continue
            if not (RMIN <= rj <= RMAX):
                continue
            err = np.abs(aj - ai) + 1.0 * np.abs(bj + bi) + np.abs(rj - ri)
            if err < best:
                best = err
                jbest = j
        if jbest == -1:
            continue
        used[i] = True
        used[jbest] = True
        a = 0.5 * (ai + lam_re[jbest])
        b = 0.5 * (bi - lam_im[jbest])
        r = np.hypot(a, b)
        th = np.arctan2(b, a)
        s_re = np.log(max(r, 1e-300)) / dt
        s_im = th / dt
        f = np.abs(s_im) / (2.0 * np.pi)
        w = 2.0 * np.pi * f
        alpha = -s_re
        zeta = 0.0
        Qv = BIGQ
        if f > 0.0 and np.isfinite(alpha):
            den = np.sqrt(alpha**2 + w**2)
            if den > 0.0:
                zeta = alpha / den
            zeta = max(0.0, min(zeta, 0.5))
            Qv = 1.0 / (2.0 * zeta) if zeta > 0.0 else BIGQ
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
    return F, Q, Z

def read_index(index_path: str):
    with open(index_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].replace('#', '').strip()
    parts = header.split()
    R = int(parts[0].split('=')[1])
    M_raw = int(parts[1].split('=')[1])
    M_out = int(parts[2].split('=')[1])
    N_use = int(parts[3].split('=')[1])
    fs = float(parts[4].split('=')[1])
    fs = abs(fs)
    names = [line.strip() for line in lines[1:] if line.strip() and not line.startswith('#')]
    if len(names) != R:
        raise ValueError("Names count mismatch")
    return R, M_raw, M_out, N_use, fs, names

def load_cube(cube_path: str, R: int, M_out: int, N_use: int) -> np.ndarray:
    flat = np.fromfile(cube_path, dtype=np.float64)
    if len(flat) != R * M_out * N_use:
        raise ValueError("Size mismatch")
    return flat.reshape((R, M_out, N_use))

esprit_ref.build_hankel = build_hankel
esprit_ref.TLS_ESPRIT_FromUs = TLS_ESPRIT_FromUs
esprit_ref.LambdasToFZQ = LambdasToFZQ
esprit_ref.read_index = read_index
esprit_ref.load_cube = load_cube
""")


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two ESPRIT implementations."""
    implementation: str
    n_modes: int
    frequencies: List[float]
    damping_ratios: List[float]
    mean_frequency: float
    std_frequency: float
    mean_damping: float
    std_damping: float
    frequency_range: Tuple[float, float]
    processing_time: float


def load_measurement_file(filepath: str, skip_channel: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load measurement file.

    Args:
        filepath: Path to measurement file
        skip_channel: Channel to skip (default: 2)

    Returns:
        force: Force channel data, shape (N,)
        responses: Response channels data, shape (N, n_channels)
    """
    data = np.loadtxt(filepath, skiprows=1)
    force = data[:, 0]

    # Skip the specified channel (e.g., channel 2 = column 3)
    if skip_channel is not None:
        response_columns = [i for i in range(1, data.shape[1]) if i != (skip_channel + 1)]
    else:
        response_columns = list(range(1, data.shape[1]))

    responses = data[:, response_columns]
    return force, responses


def detect_contact_end(force: np.ndarray, tail_fraction: float = 0.03) -> int:
    """Detect end of contact from force signal."""
    threshold = np.max(np.abs(force)) * tail_fraction
    contact_mask = np.abs(force) > threshold
    if np.any(contact_mask):
        return np.where(contact_mask)[0][-1]
    return 100  # Default


def preprocess_signals(force: np.ndarray, responses: np.ndarray, fs: float) -> np.ndarray:
    """Simple preprocessing: remove contact period and apply highpass filter."""
    contact_end = detect_contact_end(force, tail_fraction=0.03)

    # Remove contact period
    signals = responses[contact_end:, :]

    # Highpass filter at 1 Hz
    sos = scipy.signal.butter(4, 1.0 / (fs / 2), btype='highpass', output='sos')
    for ch in range(signals.shape[1]):
        signals[:, ch] = scipy.signal.sosfilt(sos, signals[:, ch])

    return signals


def preprocess_signal_band(signals: np.ndarray, fs: float,
                           low_freq: float, high_freq: float,
                           decimate_factor: int = 1,
                           exp_alpha: float = 0.0,
                           filter_order: int = 6) -> Tuple[np.ndarray, float]:
    """
    Preprocess signals for a frequency band (matching esprit.py preprocessing).

    Args:
        signals: Multi-channel signals, shape (N_samples, n_channels)
        fs: Sampling frequency
        low_freq, high_freq: Band limits
        decimate_factor: Decimation factor
        exp_alpha: Exponential window parameter
        filter_order: Butterworth filter order

    Returns:
        processed: Processed signals, shape (N_proc, n_channels)
        fs_proc: Processed sampling frequency
    """
    N_use, n_channels = signals.shape

    # Bandpass filter
    sos = scipy.signal.butter(filter_order,
                             [low_freq / (fs / 2), high_freq / (fs / 2)],
                             btype='bandpass', analog=False, output='sos')

    processed = np.zeros_like(signals)
    for ch in range(n_channels):
        processed[:, ch] = scipy.signal.sosfilt(sos, signals[:, ch])

    # Exponential window
    if exp_alpha > 0:
        t = np.arange(N_use) / fs
        window = np.exp(-exp_alpha * t)
        processed *= window[:, np.newaxis]

    # Decimation
    if decimate_factor > 1:
        # Decimate each channel and get actual output length
        decimated_channels = []
        for ch in range(n_channels):
            dec_ch = scipy.signal.decimate(processed[:, ch],
                                           decimate_factor,
                                           ftype='iir')
            decimated_channels.append(dec_ch)

        # Find minimum length (in case they differ slightly)
        min_len = min(len(ch) for ch in decimated_channels)

        # Truncate all to same length
        processed = np.column_stack([ch[:min_len] for ch in decimated_channels])
        fs_proc = fs / decimate_factor
    else:
        fs_proc = fs

    return processed, fs_proc


def run_esprit_core(signals: np.ndarray, fs: float,
                    window_length: int, model_order: int,
                    use_stabilization: bool = True,
                    use_tls: bool = False) -> ComparisonMetrics:
    """Run esprit_core.py implementation."""
    import time

    start_time = time.time()

    # Use esprit_modal_identification with appropriate flags
    result = esprit_core.esprit_modal_identification(
        signals=signals,
        fs=fs,
        window_length=window_length,
        model_order=model_order,
        use_stabilization=use_stabilization,
        use_tls=use_tls,
        use_gpu=False,
        ref_sensor=0,
        max_damping=0.2,
        freq_range=(0, np.inf)
    )

    processing_time = time.time() - start_time

    # Extract metrics
    freqs = result.frequencies.tolist()
    damps = result.damping_ratios.tolist()

    return ComparisonMetrics(
        implementation="esprit_core" + (" (TLS)" if use_tls else " (LS)") +
                      (" + stab" if use_stabilization else ""),
        n_modes=len(freqs),
        frequencies=freqs,
        damping_ratios=damps,
        mean_frequency=np.mean(freqs) if freqs else 0.0,
        std_frequency=np.std(freqs) if freqs else 0.0,
        mean_damping=np.mean(damps) if damps else 0.0,
        std_damping=np.std(damps) if damps else 0.0,
        frequency_range=(min(freqs), max(freqs)) if freqs else (0.0, 0.0),
        processing_time=processing_time
    )


def run_esprit_ref(signals: np.ndarray, fs: float,
                   window_length: int, model_order: int) -> ComparisonMetrics:
    """Run esprit.py reference implementation (TLS-ESPRIT from Us)."""
    import time

    start_time = time.time()

    N_band, n_channels = signals.shape
    dt = 1.0 / fs
    L = window_length

    # Build multi-channel Hankel (matching esprit.py approach)
    big_H = np.vstack([esprit_ref.build_hankel(signals[:, ch], N_band, L)
                       for ch in range(n_channels)])

    # SVD
    U, S, Vh = np.linalg.svd(big_H, full_matrices=False)

    # TLS-ESPRIT from Us
    Us = U[:, :model_order]
    lam_re, lam_im = esprit_ref.TLS_ESPRIT_FromUs(Us, n_channels * L, L,
                                                   n_channels, model_order)

    # Convert to frequencies and damping
    F, Q, Z = esprit_ref.LambdasToFZQ(lam_re, lam_im, model_order, dt)

    processing_time = time.time() - start_time

    # Extract metrics
    freqs = F.tolist()
    damps = Z.tolist()

    return ComparisonMetrics(
        implementation="esprit.py (TLS-U)",
        n_modes=len(freqs),
        frequencies=freqs,
        damping_ratios=damps,
        mean_frequency=np.mean(freqs) if freqs else 0.0,
        std_frequency=np.std(freqs) if freqs else 0.0,
        mean_damping=np.mean(damps) if damps else 0.0,
        std_damping=np.std(damps) if damps else 0.0,
        frequency_range=(min(freqs), max(freqs)) if freqs else (0.0, 0.0),
        processing_time=processing_time
    )


def visualize_single_point_comparison(metrics_list: List[ComparisonMetrics],
                                     signals: np.ndarray, fs: float,
                                     point_name: str,
                                     save_path: Optional[str] = None):
    """
    Visualize single-point comparison with side-by-side results.

    Creates a comprehensive figure with:
    - Time series plot
    - Frequency spectrum
    - Mode identification comparison (frequency vs damping)
    - Metrics table
    """
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Time series (top row, spans 2 columns)
    ax_time = fig.add_subplot(gs[0, :2])
    t = np.arange(len(signals)) / fs
    ax_time.plot(t, signals[:, 0], 'b-', linewidth=0.5, alpha=0.7)
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Amplitude')
    ax_time.set_title(f'Time Series - {point_name} (Channel 0)')
    ax_time.grid(True, alpha=0.3)

    # FFT (top right, spans 2 columns)
    ax_fft = fig.add_subplot(gs[0, 2:])
    fft_vals = np.fft.rfft(signals[:, 0])
    fft_freqs = np.fft.rfftfreq(len(signals), 1/fs)
    ax_fft.semilogy(fft_freqs, np.abs(fft_vals), 'b-', linewidth=0.5)
    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Magnitude')
    ax_fft.set_title('Frequency Spectrum')
    ax_fft.grid(True, alpha=0.3)
    ax_fft.set_xlim([0, fs/2])

    # Mode comparison (middle row) - one subplot per implementation
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Determine common frequency and damping ranges across all implementations
    all_freqs = []
    all_damps = []
    for m in metrics_list:
        if m.frequencies:
            all_freqs.extend(m.frequencies)
            all_damps.extend(m.damping_ratios)

    if all_freqs:
        freq_min = min(all_freqs)
        freq_max = max(all_freqs)
        freq_range = freq_max - freq_min
        freq_min = max(0, freq_min - 0.1 * freq_range)
        freq_max = freq_max + 0.1 * freq_range

        damp_min = 0
        damp_max = max(all_damps) * 100 * 1.2  # Convert to % and add margin
        damp_max = max(damp_max, 20)  # At least 20%
    else:
        freq_min, freq_max = 0, 500
        damp_min, damp_max = 0, 20

    for idx, metrics in enumerate(metrics_list):
        ax_modes = fig.add_subplot(gs[1, idx])

        if metrics.frequencies:
            freqs = np.array(metrics.frequencies)
            damps = np.array(metrics.damping_ratios) * 100  # Convert to %

            # Sort modes by frequency for consistent numbering
            sort_idx = np.argsort(freqs)
            freqs = freqs[sort_idx]
            damps = damps[sort_idx]

            ax_modes.scatter(freqs, damps, s=80, alpha=0.7,
                           color=colors[idx % len(colors)],
                           edgecolors='black', linewidth=1.5)

            # Add frequency labels
            for f, d in zip(freqs, damps):
                ax_modes.annotate(f'{f:.1f}', (f, d),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=7, alpha=0.7)

        ax_modes.set_xlabel('Frequency (Hz)')
        ax_modes.set_ylabel('Damping Ratio (%)')
        ax_modes.set_title(f'{metrics.implementation}\n({metrics.n_modes} modes)')
        ax_modes.grid(True, alpha=0.3)
        # Use common ranges for all subplots
        ax_modes.set_xlim([freq_min, freq_max])
        ax_modes.set_ylim([damp_min, damp_max])

    # Frequency comparison histogram (bottom left)
    ax_hist = fig.add_subplot(gs[2, 0])
    for idx, metrics in enumerate(metrics_list):
        if metrics.frequencies:
            ax_hist.hist(metrics.frequencies, bins=20, alpha=0.5,
                        label=metrics.implementation,
                        color=colors[idx % len(colors)])
    ax_hist.set_xlabel('Frequency (Hz)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Frequency Distribution')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Damping comparison histogram (bottom middle)
    ax_damp_hist = fig.add_subplot(gs[2, 1])
    for idx, metrics in enumerate(metrics_list):
        if metrics.damping_ratios:
            ax_damp_hist.hist(np.array(metrics.damping_ratios) * 100,
                             bins=20, alpha=0.5,
                             label=metrics.implementation,
                             color=colors[idx % len(colors)])
    ax_damp_hist.set_xlabel('Damping Ratio (%)')
    ax_damp_hist.set_ylabel('Count')
    ax_damp_hist.set_title('Damping Distribution')
    ax_damp_hist.legend(fontsize=8)
    ax_damp_hist.grid(True, alpha=0.3)

    # Metrics table (bottom right, spans 2 columns)
    ax_table = fig.add_subplot(gs[2, 2:])
    ax_table.axis('off')

    table_data = []
    headers = ['Implementation', 'Modes', 'Time (s)']
    for metrics in metrics_list:
        table_data.append([
            metrics.implementation.replace(' ', '\n'),
            str(metrics.n_modes),
            f'{metrics.processing_time:.3f}'
        ])

    table = ax_table.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          colWidths=[0.5, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    plt.suptitle(f'Single-Point Analysis Comparison - {point_name}',
                fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved single-point comparison to {save_path}")

    plt.show()


def visualize_multipoint_comparison(all_metrics: Dict[str, List[ComparisonMetrics]],
                                   point_names: List[str],
                                   save_path: Optional[str] = None):
    """
    Visualize multi-point comparison (single frequency band).

    Creates stabilization diagrams and consistency analysis.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Get implementation names
    impl_names = list(all_metrics.keys())

    # Stabilization diagrams for each implementation
    for idx, impl_name in enumerate(impl_names[:3]):
        ax = fig.add_subplot(gs[0, idx])

        # Collect all frequencies and point indices
        all_freqs = []
        all_points = []

        for pt_idx, metrics in enumerate(all_metrics[impl_name]):
            if metrics.frequencies:
                all_freqs.extend(metrics.frequencies)
                all_points.extend([pt_idx] * len(metrics.frequencies))

        if all_freqs:
            ax.scatter(all_points, all_freqs, s=20, alpha=0.6,
                      color=colors[idx % len(colors)])
            ax.set_xlabel('Point Index')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Stabilization Diagram\n{impl_name}')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(len(point_names)))
            ax.set_xticklabels([f'P{i}' for i in range(len(point_names))],
                              rotation=45, fontsize=8)

    # Frequency consistency analysis (bottom left)
    ax_consistency = fig.add_subplot(gs[1, 0])

    for idx, impl_name in enumerate(impl_names):
        # Bin frequencies and count occurrences across points
        all_freqs = []
        for metrics in all_metrics[impl_name]:
            all_freqs.extend(metrics.frequencies)

        if all_freqs:
            hist, bins = np.histogram(all_freqs, bins=30)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax_consistency.plot(bin_centers, hist, marker='o',
                              label=impl_name, alpha=0.7,
                              color=colors[idx % len(colors)])

    ax_consistency.set_xlabel('Frequency (Hz)')
    ax_consistency.set_ylabel('Occurrence Count')
    ax_consistency.set_title('Frequency Consistency Across Points')
    ax_consistency.legend(fontsize=8)
    ax_consistency.grid(True, alpha=0.3)

    # Mode count comparison (bottom middle)
    ax_mode_count = fig.add_subplot(gs[1, 1])

    x_pos = np.arange(len(point_names))
    width = 0.8 / len(impl_names)

    for idx, impl_name in enumerate(impl_names):
        mode_counts = [m.n_modes for m in all_metrics[impl_name]]
        ax_mode_count.bar(x_pos + idx * width, mode_counts, width,
                         label=impl_name, alpha=0.7,
                         color=colors[idx % len(colors)])

    ax_mode_count.set_xlabel('Point Index')
    ax_mode_count.set_ylabel('Number of Modes')
    ax_mode_count.set_title('Mode Count per Point')
    ax_mode_count.set_xticks(x_pos + width * (len(impl_names) - 1) / 2)
    ax_mode_count.set_xticklabels([f'P{i}' for i in range(len(point_names))],
                                  rotation=45, fontsize=8)
    ax_mode_count.legend(fontsize=8)
    ax_mode_count.grid(True, alpha=0.3, axis='y')

    # Statistics table (bottom right)
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis('off')

    table_data = []
    headers = ['Implementation', 'Avg\nModes', 'Total\nModes', 'Avg\nFreq (Hz)']

    for impl_name in impl_names:
        metrics_list = all_metrics[impl_name]
        avg_modes = np.mean([m.n_modes for m in metrics_list])
        total_modes = sum([m.n_modes for m in metrics_list])
        all_freqs = []
        for m in metrics_list:
            all_freqs.extend(m.frequencies)
        avg_freq = np.mean(all_freqs) if all_freqs else 0.0

        table_data.append([
            impl_name.replace(' ', '\n'),
            f'{avg_modes:.1f}',
            str(total_modes),
            f'{avg_freq:.1f}'
        ])

    table = ax_stats.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    plt.suptitle('Multi-Point Analysis Comparison (Single Band)',
                fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-point comparison to {save_path}")

    plt.show()


def analyze_single_point(filepath: str, fs: float, band_config: Dict,
                        window_length: int, model_order: int) -> Tuple[List[ComparisonMetrics], np.ndarray, float]:
    """
    Run single-point analysis with all implementations.

    Returns list of metrics for comparison, signals, and processed fs.
    """
    point_name = Path(filepath).stem

    print(f"\n{'='*60}")
    print(f"SINGLE-POINT ANALYSIS: {point_name}")
    print(f"{'='*60}")

    # Load and preprocess
    force, responses = load_measurement_file(filepath, skip_channel=2)
    signals_raw = preprocess_signals(force, responses, fs)

    # Preprocess for band
    signals, fs_proc = preprocess_signal_band(
        signals_raw, fs,
        low_freq=band_config['low_freq'],
        high_freq=band_config['high_freq'],
        decimate_factor=band_config['decimate_factor'],
        exp_alpha=band_config['exp_alpha'],
        filter_order=band_config['filter_order']
    )

    print(f"Processed: {signals.shape[0]} samples, {signals.shape[1]} channels, fs={fs_proc:.1f} Hz")
    print(f"Band: {band_config['low_freq']}-{band_config['high_freq']} Hz")

    metrics_list = []

    # 1. esprit.py reference (TLS-U)
    print("\n[1/4] Running esprit.py (TLS-U reference)...")
    metrics_ref = run_esprit_ref(signals, fs_proc, window_length, model_order)
    metrics_list.append(metrics_ref)
    print(f"  -> {metrics_ref.n_modes} modes, {metrics_ref.processing_time:.3f}s")

    # 2. esprit_core LS
    print("\n[2/4] Running esprit_core (LS, no stabilization)...")
    metrics_core_ls = run_esprit_core(signals, fs_proc, window_length, model_order,
                                      use_stabilization=False, use_tls=False)
    metrics_list.append(metrics_core_ls)
    print(f"  -> {metrics_core_ls.n_modes} modes, {metrics_core_ls.processing_time:.3f}s")

    # 3. esprit_core TLS
    print("\n[3/4] Running esprit_core (TLS, no stabilization)...")
    metrics_core_tls = run_esprit_core(signals, fs_proc, window_length, model_order,
                                       use_stabilization=False, use_tls=True)
    metrics_list.append(metrics_core_tls)
    print(f"  -> {metrics_core_tls.n_modes} modes, {metrics_core_tls.processing_time:.3f}s")

    # 4. esprit_core LS + Stabilization
    print("\n[4/4] Running esprit_core (LS + stabilization)...")
    metrics_core_stab = run_esprit_core(signals, fs_proc, window_length, model_order,
                                        use_stabilization=True, use_tls=False)
    metrics_list.append(metrics_core_stab)
    print(f"  -> {metrics_core_stab.n_modes} modes, {metrics_core_stab.processing_time:.3f}s")

    # Print summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    for metrics in metrics_list:
        print(f"{metrics.implementation:30s}: {metrics.n_modes:3d} modes, "
              f"f=[{metrics.frequency_range[0]:6.1f}, {metrics.frequency_range[1]:6.1f}] Hz, "
              f"time={metrics.processing_time:.3f}s")

    return metrics_list, signals, fs_proc


def analyze_multipoint_single_band(data_files: List[Path], fs: float,
                                   band_config: Dict, window_length: int,
                                   model_order: int) -> Dict[str, List[ComparisonMetrics]]:
    """
    Run multi-point analysis for single frequency band.

    Returns dict mapping implementation name to list of metrics (one per point).
    """
    print(f"\n{'='*60}")
    print(f"MULTI-POINT SINGLE-BAND ANALYSIS")
    print(f"Band: {band_config['low_freq']}-{band_config['high_freq']} Hz")
    print(f"Points: {len(data_files)}")
    print(f"{'='*60}")

    all_metrics = {
        'esprit.py (TLS-U)': [],
        'esprit_core (LS)': [],
        'esprit_core (TLS)': [],
        'esprit_core (LS+stab)': []
    }

    for filepath in data_files:
        pt_name = filepath.stem
        print(f"\nProcessing point {pt_name}...")

        # Load and preprocess
        force, responses = load_measurement_file(str(filepath), skip_channel=2)
        signals_raw = preprocess_signals(force, responses, fs)

        signals, fs_proc = preprocess_signal_band(
            signals_raw, fs,
            low_freq=band_config['low_freq'],
            high_freq=band_config['high_freq'],
            decimate_factor=band_config['decimate_factor'],
            exp_alpha=band_config['exp_alpha'],
            filter_order=band_config['filter_order']
        )

        # Run all implementations
        metrics_ref = run_esprit_ref(signals, fs_proc, window_length, model_order)
        metrics_core_ls = run_esprit_core(signals, fs_proc, window_length, model_order,
                                         use_stabilization=False, use_tls=False)
        metrics_core_tls = run_esprit_core(signals, fs_proc, window_length, model_order,
                                          use_stabilization=False, use_tls=True)
        metrics_core_stab = run_esprit_core(signals, fs_proc, window_length, model_order,
                                           use_stabilization=True, use_tls=False)

        all_metrics['esprit.py (TLS-U)'].append(metrics_ref)
        all_metrics['esprit_core (LS)'].append(metrics_core_ls)
        all_metrics['esprit_core (TLS)'].append(metrics_core_tls)
        all_metrics['esprit_core (LS+stab)'].append(metrics_core_stab)

        print(f"  esprit.py: {metrics_ref.n_modes} modes")
        print(f"  core (LS): {metrics_core_ls.n_modes} modes")
        print(f"  core (TLS): {metrics_core_tls.n_modes} modes")
        print(f"  core (LS+stab): {metrics_core_stab.n_modes} modes")

    return all_metrics


def save_comparison_results(results: Dict, save_path: str):
    """Save comparison results to JSON."""
    # Convert metrics to dicts
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, list):
            results_serializable[key] = [asdict(m) if hasattr(m, '__dataclass_fields__')
                                        else m for m in value]
        elif isinstance(value, dict):
            results_serializable[key] = {
                k: [asdict(m) if hasattr(m, '__dataclass_fields__') else m for m in v]
                for k, v in value.items()
            }
        else:
            results_serializable[key] = value

    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nSaved results to {save_path}")


def main():
    """Main comparison workflow."""
    # Configuration
    DATA_DIR = Path("piano_point_responses")
    OUTPUT_DIR = Path("ESPRIT/comparison_results")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Band configuration (use band 0 from esprit.py)
    BAND_CONFIG = {
        'low_freq': 40.0,
        'high_freq': 500.0,
        'decimate_factor': 4,
        'exp_alpha': 0.01,
        'filter_order': 6
    }

    # ESPRIT parameters
    WINDOW_LENGTH = 1024  # L
    MODEL_ORDER = 30      # K
    FS = 20000.0          # Sampling frequency

    # Load data files
    print("Loading data...")
    data_files = sorted(list(DATA_DIR.glob("point_*_response.txt")))
    print(f"Found {len(data_files)} measurement files")

    # ========================================================================
    # LEVEL 1: Single-point analysis
    # ========================================================================
    SINGLE_POINT_FILE = data_files[0]  # First point

    metrics_single, signals_single, fs_single = analyze_single_point(
        str(SINGLE_POINT_FILE), FS, BAND_CONFIG, WINDOW_LENGTH, MODEL_ORDER
    )

    visualize_single_point_comparison(
        metrics_single, signals_single, fs_single, SINGLE_POINT_FILE.stem,
        save_path=OUTPUT_DIR / "comparison_single_point.png"
    )

    # ========================================================================
    # LEVEL 2: Multi-point single-band analysis
    # ========================================================================
    all_metrics_multipoint = analyze_multipoint_single_band(
        data_files, FS, BAND_CONFIG, WINDOW_LENGTH, MODEL_ORDER
    )

    point_names = [f.stem for f in data_files]
    visualize_multipoint_comparison(
        all_metrics_multipoint, point_names,
        save_path=OUTPUT_DIR / "comparison_multipoint_single_band.png"
    )

    # ========================================================================
    # LEVEL 3: Full dataset statistical comparison
    # ========================================================================
    print(f"\n{'='*60}")
    print("FULL DATASET STATISTICAL COMPARISON")
    print(f"{'='*60}")

    # Aggregate statistics
    stats_summary = {}
    for impl_name, metrics_list in all_metrics_multipoint.items():
        all_freqs = []
        all_damps = []
        for m in metrics_list:
            all_freqs.extend(m.frequencies)
            all_damps.extend(m.damping_ratios)

        stats_summary[impl_name] = {
            'total_modes': sum([m.n_modes for m in metrics_list]),
            'avg_modes_per_point': np.mean([m.n_modes for m in metrics_list]),
            'std_modes_per_point': np.std([m.n_modes for m in metrics_list]),
            'mean_frequency': np.mean(all_freqs) if all_freqs else 0.0,
            'std_frequency': np.std(all_freqs) if all_freqs else 0.0,
            'mean_damping_pct': np.mean(all_damps) * 100 if all_damps else 0.0,
            'std_damping_pct': np.std(all_damps) * 100 if all_damps else 0.0,
            'total_time': sum([m.processing_time for m in metrics_list])
        }

    # Print summary
    print(f"\n{'Implementation':<25} {'Total':>8} {'Avg':>8} {'Std':>8} "
          f"{'AvgFreq':>8} {'AvgDamp%':>8} {'Time(s)':>8}")
    print('-' * 80)
    for impl_name, stats in stats_summary.items():
        print(f"{impl_name:<25} {stats['total_modes']:8d} "
              f"{stats['avg_modes_per_point']:8.1f} "
              f"{stats['std_modes_per_point']:8.1f} "
              f"{stats['mean_frequency']:8.1f} "
              f"{stats['mean_damping_pct']:8.2f} "
              f"{stats['total_time']:8.2f}")

    # Save all results
    save_comparison_results({
        'single_point': {
            'point_name': SINGLE_POINT_FILE.stem,
            'metrics': metrics_single
        },
        'multipoint': all_metrics_multipoint,
        'statistics': stats_summary,
        'configuration': {
            'band': BAND_CONFIG,
            'window_length': WINDOW_LENGTH,
            'model_order': MODEL_ORDER,
            'fs': FS
        }
    }, OUTPUT_DIR / "comparison_results.json")

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
