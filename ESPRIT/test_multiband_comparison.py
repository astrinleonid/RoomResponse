"""
Multi-band ESPRIT comparison test.

Compares single-band vs multi-band approaches for modal identification
across wide frequency range (40-4000 Hz).

Multi-band approach (matching esprit.py):
- Band 1: 40-500 Hz (decimate x4)
- Band 2: 500-1000 Hz (decimate x2)
- Band 3: 1000-2000 Hz (no decimation)
- Band 4: 2000-4000 Hz (no decimation)
"""
import numpy as np
import time
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import esprit_core


class BandConfig:
    """Configuration for a single frequency band."""
    def __init__(self, low_freq: float, high_freq: float,
                 decimate_factor: int, filter_order: int,
                 exp_alpha: float, model_order: int):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.decimate_factor = decimate_factor
        self.filter_order = filter_order
        self.exp_alpha = exp_alpha
        self.model_order = model_order


# Band presets matching esprit.py
BAND_CONFIGS = [
    BandConfig(40.0, 500.0, 4, 6, 0.01, 20),
    BandConfig(500.0, 1000.0, 2, 6, 0.005, 15),
    BandConfig(1000.0, 2000.0, 1, 8, 0.002, 15),
    BandConfig(2000.0, 4000.0, 1, 8, 0.001, 15),
]


def process_single_band(signal_data: np.ndarray, fs: float,
                       band_config: BandConfig) -> dict:
    """
    Process a single frequency band.

    Args:
        signal_data: Input signal, shape (N,) or (N, n_channels)
        fs: Sampling frequency (Hz)
        band_config: Band configuration

    Returns:
        dict with frequencies, damping_ratios, and band info
    """
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(-1, 1)

    N, n_channels = signal_data.shape

    # Extract first channel for processing
    y = signal_data[:, 0].copy()

    # Apply bandpass filter
    nyq = fs / 2
    # Ensure critical frequencies are within valid range (0, 1)
    low_norm = max(band_config.low_freq / nyq, 0.01)  # Minimum 1% of Nyquist
    high_norm = min(band_config.high_freq / nyq, 0.99)  # Maximum 99% of Nyquist

    if high_norm <= low_norm:
        print(f"    Warning: Band {band_config.low_freq:.0f}-{band_config.high_freq:.0f} Hz exceeds Nyquist, skipping")
        return {
            'frequencies': np.array([]),
            'damping_ratios': np.array([]),
            'n_modes': 0,
            'band_range': (band_config.low_freq, band_config.high_freq),
            'fs_band': fs,
            'decimate_factor': band_config.decimate_factor
        }

    sos = signal.butter(
        band_config.filter_order,
        [low_norm, high_norm],
        btype='bandpass',
        output='sos'
    )
    y_band = signal.sosfilt(sos, y)

    # Apply exponential window
    t = np.arange(len(y_band)) / fs
    y_band *= np.exp(-band_config.exp_alpha * t)

    # Decimate if needed
    fs_band = fs
    if band_config.decimate_factor > 1:
        y_band = signal.decimate(y_band, band_config.decimate_factor, ftype='iir')
        fs_band = fs / band_config.decimate_factor

    # Prepare for ESPRIT
    N_band = len(y_band)
    signals_band = y_band.reshape(-1, 1)

    # Run ESPRIT on this band
    window_length = min(N_band // 2, 2000)

    try:
        result = esprit_core.esprit_modal_identification(
            signals=signals_band,
            fs=fs_band,
            model_order=band_config.model_order,
            window_length=window_length,
            use_tls=True,
            use_stabilization=False,
            use_conjugate_pairing=True,
            min_freq=band_config.low_freq,
            freq_range=(band_config.low_freq, band_config.high_freq),
            max_damping=0.5
        )

        return {
            'frequencies': result.frequencies,
            'damping_ratios': result.damping_ratios,
            'n_modes': len(result.frequencies),
            'band_range': (band_config.low_freq, band_config.high_freq),
            'fs_band': fs_band,
            'decimate_factor': band_config.decimate_factor
        }
    except Exception as e:
        print(f"    Band {band_config.low_freq:.0f}-{band_config.high_freq:.0f} Hz error: {e}")
        return {
            'frequencies': np.array([]),
            'damping_ratios': np.array([]),
            'n_modes': 0,
            'band_range': (band_config.low_freq, band_config.high_freq),
            'fs_band': fs_band,
            'decimate_factor': band_config.decimate_factor
        }


def multiband_esprit(signal_data: np.ndarray, fs: float,
                    band_configs: list) -> dict:
    """
    Multi-band ESPRIT processing.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        band_configs: List of BandConfig objects

    Returns:
        Combined results from all bands
    """
    all_frequencies = []
    all_damping = []
    band_results = []

    for band_config in band_configs:
        print(f"  Processing band {band_config.low_freq:.0f}-{band_config.high_freq:.0f} Hz "
              f"(decimate x{band_config.decimate_factor})...")

        band_result = process_single_band(signal_data, fs, band_config)
        band_results.append(band_result)

        if band_result['n_modes'] > 0:
            all_frequencies.extend(band_result['frequencies'])
            all_damping.extend(band_result['damping_ratios'])
            print(f"    Found {band_result['n_modes']} modes")
        else:
            print(f"    Found 0 modes")

    # Combine and sort by frequency
    if len(all_frequencies) > 0:
        all_frequencies = np.array(all_frequencies)
        all_damping = np.array(all_damping)

        sort_idx = np.argsort(all_frequencies)
        all_frequencies = all_frequencies[sort_idx]
        all_damping = all_damping[sort_idx]
    else:
        all_frequencies = np.array([])
        all_damping = np.array([])

    return {
        'frequencies': all_frequencies,
        'damping_ratios': all_damping,
        'n_modes': len(all_frequencies),
        'band_results': band_results
    }


def generate_test_signal(fs: float, duration: float, modes: list) -> np.ndarray:
    """Generate synthetic signal with known modes."""
    N = int(fs * duration)
    t = np.arange(N) / fs

    x = np.zeros(N)

    for f, zeta, A, ph in modes:
        omega_n = 2.0 * np.pi * f
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        alpha = zeta * omega_n

        x += A * np.exp(-alpha * t) * np.cos(omega_d * t + ph)

    # Add small noise
    x += np.random.randn(N) * 0.01 * np.std(x)

    return x


def run_comparison_test():
    """Run single-band vs multi-band comparison."""

    print("="*80)
    print("MULTI-BAND ESPRIT COMPARISON TEST")
    print("="*80)

    # Generate test signal with modes across full range
    fs = 8000.0
    duration = 1.0

    # Create modes distributed across frequency range
    np.random.seed(42)
    test_modes = []

    # Low frequency modes (40-500 Hz)
    for f in [50, 100, 150, 200, 300, 400]:
        zeta = 0.005 + 0.01 * np.random.rand()
        A = 1.0 / (1 + (f / 200)**2)
        ph = np.random.rand() * 2 * np.pi
        test_modes.append((f, zeta, A, ph))

    # Mid frequency modes (500-2000 Hz)
    for f in [600, 800, 1000, 1200, 1500, 1800]:
        zeta = 0.01 + 0.02 * np.random.rand()
        A = 0.5 / (1 + (f / 1000)**2)
        ph = np.random.rand() * 2 * np.pi
        test_modes.append((f, zeta, A, ph))

    # High frequency modes (2000-4000 Hz)
    for f in [2200, 2600, 3000, 3500]:
        zeta = 0.02 + 0.03 * np.random.rand()
        A = 0.3 / (1 + (f / 2500)**2)
        ph = np.random.rand() * 2 * np.pi
        test_modes.append((f, zeta, A, ph))

    print(f"\nTest Configuration:")
    print(f"  Total modes: {len(test_modes)}")
    print(f"  Frequency range: {min(m[0] for m in test_modes):.0f} - {max(m[0] for m in test_modes):.0f} Hz")
    print(f"  Signal: {duration} s @ {fs:.0f} Hz")

    # Generate signal
    signal_data = generate_test_signal(fs, duration, test_modes)

    # Test 1: Single-band processing (current esprit_core approach)
    print(f"\n{'='*80}")
    print("[1/2] Single-band processing (esprit_core standard)...")
    print("="*80)

    t0 = time.perf_counter()

    window_length = len(signal_data) // 2
    result_single = esprit_core.esprit_modal_identification(
        signals=signal_data.reshape(-1, 1),
        fs=fs,
        model_order=60,
        window_length=window_length,
        use_tls=True,
        use_stabilization=False,
        use_conjugate_pairing=True,
        min_freq=40.0,
        freq_range=(40, 4000),
        max_damping=0.5
    )

    time_single = time.perf_counter() - t0

    print(f"  Detected {len(result_single.frequencies)} modes in {time_single*1000:.1f} ms")
    if len(result_single.frequencies) > 0:
        print(f"  Frequency range: {result_single.frequencies.min():.1f} - {result_single.frequencies.max():.1f} Hz")

    # Test 2: Multi-band processing
    print(f"\n{'='*80}")
    print("[2/2] Multi-band processing (4 bands)...")
    print("="*80)

    t0 = time.perf_counter()
    result_multi = multiband_esprit(signal_data, fs, BAND_CONFIGS)
    time_multi = time.perf_counter() - t0

    print(f"\n  Total detected: {result_multi['n_modes']} modes in {time_multi*1000:.1f} ms")
    if result_multi['n_modes'] > 0:
        print(f"  Frequency range: {result_multi['frequencies'].min():.1f} - {result_multi['frequencies'].max():.1f} Hz")

    # Match modes to ground truth
    true_freqs = np.array([f for f, _, _, _ in test_modes])

    def count_matches(detected_freqs, true_freqs, tol=5.0):
        matched = 0
        for f_true in true_freqs:
            if np.any(np.abs(detected_freqs - f_true) < tol):
                matched += 1
        return matched

    single_matches = count_matches(result_single.frequencies, true_freqs)
    multi_matches = count_matches(result_multi['frequencies'], true_freqs)

    # Results summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Method':<25} {'Detected':<12} {'Matched':<12} {'Match Rate':<12} {'Time (ms)':<12}")
    print("-"*80)
    print(f"{'Single-band':<25} {len(result_single.frequencies):<12} {single_matches:<12} "
          f"{single_matches/len(test_modes)*100:<12.1f}% {time_single*1000:<12.1f}")
    print(f"{'Multi-band (4 bands)':<25} {result_multi['n_modes']:<12} {multi_matches:<12} "
          f"{multi_matches/len(test_modes)*100:<12.1f}% {time_multi*1000:<12.1f}")

    print(f"\nMode detection by frequency range:")
    print("-"*80)

    # Analyze by frequency range
    for band_result in result_multi['band_results']:
        low, high = band_result['band_range']
        n_true_in_band = np.sum((true_freqs >= low) & (true_freqs < high))
        n_detected = band_result['n_modes']
        print(f"  {low:>4.0f}-{high:<4.0f} Hz: {n_detected:>2} detected / {n_true_in_band:>2} true  "
              f"(decimate x{band_result['decimate_factor']})")

    # Visualization
    create_comparison_plots(test_modes, result_single, result_multi, signal_data, fs)

    print(f"\n{'='*80}")
    print("[OK] Multi-band comparison completed!")
    print("="*80)


def create_comparison_plots(true_modes, result_single, result_multi, signal_data, fs):
    """Create visualization comparing single-band vs multi-band."""

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    true_freqs = np.array([f for f, _, _, _ in true_modes])
    true_freqs_sorted = np.sort(true_freqs)

    # FFT Spectrum
    ax_fft = fig.add_subplot(gs[0, :])
    from scipy.fft import rfft, rfftfreq
    fft_vals = rfft(signal_data)
    fft_freqs = rfftfreq(len(signal_data), 1/fs)

    ax_fft.semilogy(fft_freqs, np.abs(fft_vals), 'b-', linewidth=0.7, alpha=0.5, label='Signal spectrum')

    # Mark true modes
    for f in true_freqs:
        ax_fft.axvline(f, color='green', linestyle='--', alpha=0.3, linewidth=1)

    # Mark band boundaries
    for band_config in BAND_CONFIGS:
        ax_fft.axvline(band_config.low_freq, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
    ax_fft.axvline(BAND_CONFIGS[-1].high_freq, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)

    ax_fft.set_xlabel('Frequency (Hz)', fontsize=11)
    ax_fft.set_ylabel('Magnitude', fontsize=11)
    ax_fft.set_title('Spectrum with Band Divisions (orange) and True Modes (green)',
                     fontsize=12, fontweight='bold')
    ax_fft.grid(True, alpha=0.3, which='both')
    ax_fft.set_xlim([0, 4400])
    ax_fft.legend(fontsize=10)

    # Frequency comparison - Single-band
    ax_single = fig.add_subplot(gs[1, 0])

    x_true = np.arange(len(true_freqs_sorted))
    ax_single.scatter(x_true, true_freqs_sorted, s=120, marker='o',
                     color='green', label='True modes', alpha=0.7,
                     edgecolors='black', linewidth=2, zorder=3)

    if len(result_single.frequencies) > 0:
        freq_sorted = np.sort(result_single.frequencies)
        x_est = np.arange(len(freq_sorted))
        ax_single.scatter(x_est, freq_sorted, s=100, marker='x',
                         color='red', label='Detected', alpha=0.8, linewidth=2.5, zorder=2)

    ax_single.set_xlabel('Mode Index', fontsize=11)
    ax_single.set_ylabel('Frequency (Hz)', fontsize=11)
    ax_single.set_title(f'Single-band: {len(result_single.frequencies)} modes detected',
                       fontsize=12, fontweight='bold')
    ax_single.legend(fontsize=10)
    ax_single.grid(True, alpha=0.3)

    # Frequency comparison - Multi-band
    ax_multi = fig.add_subplot(gs[1, 1])

    ax_multi.scatter(x_true, true_freqs_sorted, s=120, marker='o',
                    color='green', label='True modes', alpha=0.7,
                    edgecolors='black', linewidth=2, zorder=3)

    if result_multi['n_modes'] > 0:
        freq_sorted = np.sort(result_multi['frequencies'])
        x_est = np.arange(len(freq_sorted))
        ax_multi.scatter(x_est, freq_sorted, s=100, marker='x',
                        color='blue', label='Detected', alpha=0.8, linewidth=2.5, zorder=2)

    ax_multi.set_xlabel('Mode Index', fontsize=11)
    ax_multi.set_ylabel('Frequency (Hz)', fontsize=11)
    ax_multi.set_title(f'Multi-band (4 bands): {result_multi["n_modes"]} modes detected',
                      fontsize=12, fontweight='bold')
    ax_multi.legend(fontsize=10)
    ax_multi.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Single-band vs Multi-band ESPRIT Comparison',
                fontsize=16, fontweight='bold')

    # Save
    output_path = Path(__file__).parent / 'test_multiband_comparison_plots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    plt.show()


if __name__ == '__main__':
    run_comparison_test()
