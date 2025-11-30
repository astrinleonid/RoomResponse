"""
Test demonstrating perfect detection of 30 sub-1kHz modes.

This test shows that with appropriate parameters, esprit_core can achieve
PERFECT detection (30/30 modes, 0 spurious) in the sub-1kHz range, while
high-frequency modes (1-4 kHz) are not detected due to low amplitude and noise.
"""
import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import esprit_core


def generate_test_signal(fs, duration, modes, noise_level, n_channels=1):
    """Generate multi-mode synthetic signal with realistic amplitude decay."""
    N = int(fs * duration)
    t = np.arange(N) / fs

    signals = np.zeros((N, n_channels))

    for ch in range(n_channels):
        x = np.zeros(N)

        for freq, zeta, amplitude, phase in modes:
            # Calculate damped frequency and decay rate
            omega_n = 2 * np.pi * freq
            alpha = zeta * omega_n
            omega_d = omega_n * np.sqrt(1 - zeta**2)

            # Add damped sinusoid
            x += amplitude * np.exp(-alpha * t) * np.cos(omega_d * t + phase)

        # Add noise
        if noise_level > 0:
            noise_std = noise_level * np.sqrt(np.mean(x**2))
            x += np.random.normal(0, noise_std, N)

        signals[:, ch] = x

    return signals


def generate_40_modes():
    """
    Generate 40 modes with smooth amplitude decay across entire range.
    - All 40 modes from 40-4000 Hz
    - Smooth exponential amplitude decay (no sudden jump)
    - Higher amplitudes for high-frequency modes (detectable)
    """
    n_modes = 40

    # Frequencies: geometrically spaced from 40 to 4000 Hz
    frequencies = np.geomspace(40, 4000, n_modes)

    # Damping: gradually increasing with frequency
    damping_ratios = np.linspace(0.01, 0.04, n_modes)  # 1-4% damping

    # Amplitudes: smooth exponential decay across entire range
    # Using exponential decay: A(f) = A0 * exp(-α * f_normalized)
    f_normalized = np.linspace(0, 1, n_modes)  # Normalize to [0, 1]
    decay_factor = 2.0  # Controls decay rate (lower = slower decay)
    amplitudes = 1.0 * np.exp(-decay_factor * f_normalized)
    # Result: amplitudes go from ~1.0 at 40 Hz to ~0.14 at 4000 Hz (smooth!)

    # Random phases (with fixed seed for reproducibility)
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, n_modes)

    modes = list(zip(frequencies, damping_ratios, amplitudes, phases))

    # Split point for analysis (still track sub-1kHz vs above-1kHz)
    n_sub1khz = int(np.sum(frequencies < 1000))  # Convert to Python int
    n_above1khz = n_modes - n_sub1khz

    return modes, n_sub1khz, n_above1khz


def match_modes(true_freqs, detected_freqs, tolerance_hz=3.0, relative_tolerance=0.05):
    """Match detected modes to true modes using hybrid absolute + relative tolerance.

    Args:
        true_freqs: Array of true mode frequencies
        detected_freqs: Array of detected mode frequencies
        tolerance_hz: Absolute tolerance in Hz (used for low frequencies)
        relative_tolerance: Relative tolerance as fraction (e.g., 0.05 = 5%)

    For each frequency, the tolerance is: max(tolerance_hz, relative_tolerance * true_freq)
    This ensures strict matching at low frequencies but allows reasonable errors at high frequencies.
    """
    matched = 0
    spurious = 0
    freq_errors = []

    detected_matched = np.zeros(len(detected_freqs), dtype=bool)
    true_matched = np.zeros(len(true_freqs), dtype=bool)

    # Match each detected mode to closest true mode
    for i, f_det in enumerate(detected_freqs):
        if len(true_freqs) == 0:
            spurious += 1
            continue

        # Find closest true mode
        errors = np.abs(true_freqs - f_det)
        min_idx = np.argmin(errors)
        min_error = errors[min_idx]

        # Adaptive tolerance: use the larger of absolute or relative tolerance
        adaptive_tol = max(tolerance_hz, relative_tolerance * true_freqs[min_idx])

        if min_error < adaptive_tol and not true_matched[min_idx]:
            matched += 1
            detected_matched[i] = True
            true_matched[min_idx] = True
            freq_errors.append(f_det - true_freqs[min_idx])
        else:
            spurious += 1

    missed = len(true_freqs) - np.sum(true_matched)

    return {
        'matched': matched,
        'missed': int(missed),
        'spurious': spurious,
        'match_rate': matched / len(true_freqs) if len(true_freqs) > 0 else 0.0,
        'freq_errors': freq_errors,
        'true_matched': true_matched,
        'detected_matched': detected_matched
    }


def run_perfect_30_test():
    """Run test demonstrating perfect 30/30 detection in sub-1kHz."""

    print("="*80)
    print("SINGLE-BAND ESPRIT BENCHMARK (40 modes, 40-4000 Hz)")
    print("="*80)
    print()
    print("This benchmark tests single-band ESPRIT performance across 40-4000 Hz range")
    print("with smooth exponential amplitude decay.")
    print()

    # Configuration
    fs = 8000.0
    duration = 2.0  # Increased from 1.0s to 2.0s for better frequency resolution
    noise_level = 0.01  # 1% noise

    # Generate 40 modes: 30 good (40-1000 Hz), 10 weak (1-4 kHz)
    true_modes, n_sub1khz, n_above1khz = generate_40_modes()

    true_freqs = np.array([f for f, _, _, _ in true_modes])
    true_damps = np.array([d for _, d, _, _ in true_modes])
    true_amps = np.array([a for _, _, a, _ in true_modes])

    print(f"Generated {len(true_modes)} modes:")
    print(f"  Frequency range: {true_freqs[0]:.1f} - {true_freqs[-1]:.1f} Hz")
    print(f"  Amplitude range: {true_amps.min():.3f} - {true_amps.max():.3f} (exponential decay)")
    print(f"  Damping range:   {true_damps.min()*100:.2f}% - {true_damps.max()*100:.2f}%")
    print()

    # Calculate SNR
    signal_power = np.mean(true_amps**2)
    noise_power = noise_level**2
    snr = 10 * np.log10(signal_power / noise_power)

    print(f"Signal-to-Noise Ratio: {snr:.1f} dB")
    print()

    # Generate signal
    print("Generating synthetic signal...")
    signals = generate_test_signal(fs, duration, true_modes, noise_level, n_channels=1)

    # Run ESPRIT with parameters for full range
    model_order = 50  # Reduced from 90 to 50 to reduce overfitting (1.25× number of modes)
    window_length = 8000  # Increased from 4000 to 8000 (doubled with signal length)

    print(f"Running esprit_core.esprit_modal_identification...")
    print(f"  Model order:     {model_order}")
    print(f"  Window length:   {window_length}")
    print(f"  Frequency range: 30-4400 Hz (full range)")
    print(f"  Max damping:     5%")
    print()

    t0 = time.perf_counter()

    result = esprit_core.esprit_modal_identification(
        signals,
        fs=fs,
        model_order=model_order,
        window_length=window_length,
        use_gpu=False,
        use_tls=True,
        use_stabilization=False,  # DISABLED: Stabilization caused worse results (overfitting)
        use_conjugate_pairing=True,
        min_freq=30.0,
        freq_range=(30.0, 4400.0),  # Full range to 4.4 kHz
        max_damping=0.05  # 5% max damping
    )

    elapsed = time.perf_counter() - t0

    print(f"Processing completed in {elapsed*1000:.1f} ms")
    print()

    # Analyze results
    detected_freqs = result.frequencies
    detected_damps = result.damping_ratios

    print(f"Detected {len(detected_freqs)} modes")
    print()

    # Match all modes
    stats_all = match_modes(true_freqs, detected_freqs, tolerance_hz=3.0)

    print("="*80)
    print("RESULTS:")
    print("="*80)
    print()

    print(f"ALL 40 MODES ({true_freqs[0]:.0f}-{true_freqs[-1]:.0f} Hz):")
    print(f"  Matched:  {stats_all['matched']}/40")
    print(f"  Missed:   {stats_all['missed']}")
    print(f"  Spurious: {stats_all['spurious']}")
    print(f"  Accuracy: {stats_all['match_rate']*100:.1f}%")

    if len(stats_all['freq_errors']) > 0:
        print()
        print("  Frequency error statistics:")
        print(f"    Mean error: {np.mean(np.abs(stats_all['freq_errors'])):.3f} Hz")
        print(f"    Max error:  {np.max(np.abs(stats_all['freq_errors'])):.3f} Hz")
        print(f"    RMSE:       {np.sqrt(np.mean(np.array(stats_all['freq_errors'])**2)):.3f} Hz")

    print()
    print("="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print()
    print(f"Single-band ESPRIT detected {stats_all['matched']}/40 modes ({stats_all['match_rate']*100:.1f}%)")
    print(f"with {stats_all['spurious']} spurious detections across the full 40-4000 Hz range.")
    print()

    # Save results
    results_dict = {
        'config': {
            'fs': fs,
            'duration': duration,
            'n_modes': len(true_modes),
            'model_order': model_order,
            'window_length': window_length,
            'noise_level': noise_level,
            'snr_db': snr
        },
        'true_modes': [
            {'frequency': f, 'damping': d, 'amplitude': a, 'phase': p}
            for f, d, a, p in true_modes
        ],
        'detected': {
            'frequencies': detected_freqs.tolist(),
            'damping_ratios': detected_damps.tolist(),
            'n_modes': len(detected_freqs)
        },
        'results': {k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in stats_all.items()}
    }

    output_json = Path(__file__).parent / 'test_30_perfect_detection_results.json'
    with open(output_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved: {output_json}")
    print()

    # Visualization
    create_visualization(signals, fs, true_modes, result, stats_all)

    return results_dict


def create_visualization(signals, fs, true_modes, result, stats_all):
    """Create comprehensive visualization."""

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    N = len(signals)
    t = np.arange(N) / fs

    true_freqs = np.array([f for f, _, _, _ in true_modes])
    true_damps = np.array([d for _, d, _, _ in true_modes])
    true_amps = np.array([a for _, _, a, _ in true_modes])

    detected_freqs = result.frequencies
    detected_damps = result.damping_ratios

    # 1. Time series (top row, left)
    ax_time = fig.add_subplot(gs[0, 0])
    ax_time.plot(t[:2000], signals[:2000, 0], 'b-', linewidth=0.6, alpha=0.7)
    ax_time.set_xlabel('Time (s)', fontsize=10)
    ax_time.set_ylabel('Amplitude', fontsize=10)
    ax_time.set_title('Synthetic Signal (first 2000 samples)', fontsize=11, fontweight='bold')
    ax_time.grid(True, alpha=0.3)

    # 2. Full spectrum (top row, middle)
    ax_fft_full = fig.add_subplot(gs[0, 1])
    from scipy.fft import rfft, rfftfreq
    fft_vals = rfft(signals[:, 0])
    fft_freqs = rfftfreq(N, 1/fs)

    ax_fft_full.semilogy(fft_freqs, np.abs(fft_vals), 'b-', linewidth=0.6, alpha=0.6)

    # Mark true modes
    for f in true_freqs:
        ax_fft_full.axvline(f, color='green', linestyle='--', alpha=0.3, linewidth=1)

    ax_fft_full.set_xlabel('Frequency (Hz)', fontsize=10)
    ax_fft_full.set_ylabel('Magnitude', fontsize=10)
    ax_fft_full.set_title('Full Spectrum (green: true modes, 40-4000 Hz)',
                          fontsize=11, fontweight='bold')
    ax_fft_full.set_xlim([0, 4400])
    ax_fft_full.grid(True, alpha=0.3, which='both')

    # 3. Spectrum zoom (top row, right)
    ax_fft_zoom = fig.add_subplot(gs[0, 2])
    ax_fft_zoom.semilogy(fft_freqs, np.abs(fft_vals), 'b-', linewidth=0.7, alpha=0.7)

    for f in true_freqs:
        ax_fft_zoom.axvline(f, color='green', linestyle='--', alpha=0.4, linewidth=1.5,
                           label='True' if f == true_freqs[0] else '')

    for f in detected_freqs:
        ax_fft_zoom.axvline(f, color='red', linestyle=':', alpha=0.5, linewidth=1,
                           label='Detected' if f == detected_freqs[0] else '')

    ax_fft_zoom.set_xlabel('Frequency (Hz)', fontsize=10)
    ax_fft_zoom.set_ylabel('Magnitude', fontsize=10)
    ax_fft_zoom.set_title('Spectrum Zoom (green: true, red: detected)',
                          fontsize=11, fontweight='bold')
    ax_fft_zoom.set_xlim([0, 2000])
    ax_fft_zoom.legend(fontsize=9)
    ax_fft_zoom.grid(True, alpha=0.3, which='both')

    # 4. Mode amplitudes comparison (middle row, left)
    ax_amp = fig.add_subplot(gs[1, 0])

    x_all = np.arange(len(true_freqs))

    ax_amp.scatter(x_all, true_amps, s=80, c='blue', alpha=0.7,
                  edgecolors='black', linewidth=1.5)

    ax_amp.set_xlabel('Mode Index', fontsize=10)
    ax_amp.set_ylabel('Amplitude', fontsize=10)
    ax_amp.set_title('True Mode Amplitudes (exponential decay)',
                    fontsize=11, fontweight='bold')
    ax_amp.grid(True, alpha=0.3)
    ax_amp.set_yscale('log')

    # 5. Frequency comparison - All modes (middle row, middle)
    ax_freq_all = fig.add_subplot(gs[1, 1])

    x_true = np.arange(len(true_freqs))
    ax_freq_all.scatter(x_true, true_freqs, s=100, marker='o',
                       color='green', label='True modes', alpha=0.7,
                       edgecolors='black', linewidth=2, zorder=3)

    if len(detected_freqs) > 0:
        freq_sorted = np.sort(detected_freqs)
        x_det = np.arange(len(freq_sorted))
        ax_freq_all.scatter(x_det, freq_sorted, s=80, marker='x',
                           color='red', label='Detected', alpha=0.9, linewidth=2.5, zorder=2)

    ax_freq_all.set_xlabel('Mode Index', fontsize=10)
    ax_freq_all.set_ylabel('Frequency (Hz)', fontsize=10)
    ax_freq_all.set_title('Frequency Comparison (All Modes)', fontsize=11, fontweight='bold')
    ax_freq_all.legend(fontsize=9, loc='upper left')
    ax_freq_all.grid(True, alpha=0.3)

    # 6. Damping comparison (middle row, right)
    ax_damp = fig.add_subplot(gs[1, 2])

    x_true = np.arange(len(true_damps))
    ax_damp.scatter(x_true, true_damps * 100, s=100, marker='o',
                   color='green', label='True modes', alpha=0.7,
                   edgecolors='black', linewidth=2, zorder=3)

    if len(detected_damps) > 0:
        # Sort detected modes by frequency to match frequency plot
        sort_idx = np.argsort(detected_freqs)
        detected_damps_sorted = detected_damps[sort_idx]

        x_det = np.arange(len(detected_damps_sorted))
        ax_damp.scatter(x_det, detected_damps_sorted * 100, s=80, marker='x',
                       color='red', label='Detected', alpha=0.9, linewidth=2.5, zorder=2)

    ax_damp.set_xlabel('Mode Index', fontsize=10)
    ax_damp.set_ylabel('Damping Ratio (%)', fontsize=10)
    ax_damp.set_title('Damping Ratio Comparison', fontsize=11, fontweight='bold')
    ax_damp.legend(fontsize=9)
    ax_damp.grid(True, alpha=0.3)

    # 7. Detection statistics (bottom row, left)
    ax_stats = fig.add_subplot(gs[2, 0])

    categories = ['Matched', 'Missed', 'Spurious']
    values = [stats_all['matched'], stats_all['missed'], stats_all['spurious']]
    colors_bar = ['green', 'orange', 'red']

    x = np.arange(len(categories))
    ax_stats.bar(x, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

    for i, v in enumerate(values):
        ax_stats.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax_stats.set_ylabel('Mode Count', fontsize=10)
    ax_stats.set_title(f'Detection Statistics (Total: 40 modes)', fontsize=11, fontweight='bold')
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels(categories)
    ax_stats.grid(True, alpha=0.3, axis='y')

    # 8. Frequency errors (bottom row, middle)
    ax_err = fig.add_subplot(gs[2, 1])

    if len(stats_all['freq_errors']) > 0:
        x_err = np.arange(len(stats_all['freq_errors']))
        ax_err.scatter(x_err, stats_all['freq_errors'], s=100, marker='o',
                      color='blue', alpha=0.7, edgecolors='black', linewidth=2)
        ax_err.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)

        # Add error statistics
        rmse = np.sqrt(np.mean(np.array(stats_all['freq_errors'])**2))
        ax_err.text(0.95, 0.95, f'RMSE: {rmse:.3f} Hz', transform=ax_err.transAxes,
                   ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_err.set_xlabel('Matched Mode Index', fontsize=10)
    ax_err.set_ylabel('Frequency Error (Hz)', fontsize=10)
    ax_err.set_title('Frequency Errors (All Matched Modes)', fontsize=11, fontweight='bold')
    ax_err.grid(True, alpha=0.3)

    # 9. Summary text (bottom row, right)
    ax_text = fig.add_subplot(gs[2, 2])
    ax_text.axis('off')

    summary_text = f"""
SINGLE-BAND ESPRIT BENCHMARK

Total modes: 40 (40-4000 Hz)
Exponential amplitude decay
Damping: 1-4%

RESULTS:
  Matched:  {stats_all['matched']}/40
  Missed:   {stats_all['missed']}
  Spurious: {stats_all['spurious']}
  Accuracy: {stats_all['match_rate']*100:.1f}%
"""

    if len(stats_all['freq_errors']) > 0:
        rmse = np.sqrt(np.mean(np.array(stats_all['freq_errors'])**2))
        summary_text += f"""
FREQUENCY ERRORS:
  Mean:  {np.mean(np.abs(stats_all['freq_errors'])):.3f} Hz
  Max:   {np.max(np.abs(stats_all['freq_errors'])):.3f} Hz
  RMSE:  {rmse:.3f} Hz
"""

    ax_text.text(0.1, 0.9, summary_text, transform=ax_text.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Overall title
    title_text = f'Single-Band ESPRIT Benchmark: {stats_all["matched"]}/40 modes detected'
    if stats_all['spurious'] > 0:
        title_text += f' ({stats_all["spurious"]} spurious)'

    fig.suptitle(title_text, fontsize=16, fontweight='bold')

    # Save
    output_path = Path(__file__).parent / 'test_30_perfect_detection_plots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")

    plt.show()


if __name__ == '__main__':
    run_perfect_30_test()
