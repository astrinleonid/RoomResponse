"""
test_esprit_core_synthetic.py
Self-test for esprit_core using synthetic multi-mode signals.

Generates artificial damped sinusoidal signals with known modal parameters,
then validates that esprit_core can accurately recover them.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

# Add ESPRIT directory to path
sys.path.insert(0, str(Path(__file__).parent))

import esprit_core


def generate_synthetic_signal(
    n_samples: int,
    fs: float,
    frequencies: np.ndarray,
    damping_ratios: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    n_channels: int = 1,
    noise_level: float = 0.0
) -> np.ndarray:
    """
    Generate synthetic multi-mode damped oscillation signal.

    Args:
        n_samples: Number of time samples
        fs: Sampling frequency (Hz)
        frequencies: Modal frequencies (Hz), shape (n_modes,)
        damping_ratios: Modal damping ratios, shape (n_modes,)
        amplitudes: Modal amplitudes, shape (n_modes,) or (n_modes, n_channels)
        phases: Modal phases (radians), shape (n_modes,)
        n_channels: Number of channels
        noise_level: Gaussian noise standard deviation

    Returns:
        signal: Synthetic signal, shape (n_samples, n_channels)
    """
    n_modes = len(frequencies)
    t = np.arange(n_samples) / fs
    dt = 1.0 / fs

    # Initialize signal
    signal = np.zeros((n_samples, n_channels))

    # Ensure amplitudes is 2D
    if amplitudes.ndim == 1:
        amplitudes = amplitudes[:, np.newaxis]  # Shape: (n_modes, 1)
        amplitudes = np.tile(amplitudes, (1, n_channels))  # Shape: (n_modes, n_channels)

    # Generate each mode
    for k in range(n_modes):
        f_k = frequencies[k]
        zeta_k = damping_ratios[k]
        A_k = amplitudes[k, :]  # Shape: (n_channels,)
        phi_k = phases[k]

        # Natural frequency and damping
        omega_n = 2 * np.pi * f_k
        omega_d = omega_n * np.sqrt(1 - zeta_k**2)  # Damped frequency
        alpha = zeta_k * omega_n  # Decay rate

        # Generate damped sinusoid for each channel
        for ch in range(n_channels):
            envelope = A_k[ch] * np.exp(-alpha * t)
            oscillation = np.cos(omega_d * t + phi_k)
            signal[:, ch] += envelope * oscillation

    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, (n_samples, n_channels))
        signal += noise

    return signal


def compare_modes(
    true_freqs: np.ndarray,
    true_damps: np.ndarray,
    est_freqs: np.ndarray,
    est_damps: np.ndarray,
    freq_tol: float = 2.0,
    damp_tol: float = 0.02
) -> dict:
    """
    Compare estimated modes to ground truth.

    Args:
        true_freqs: True frequencies (Hz)
        true_damps: True damping ratios
        est_freqs: Estimated frequencies (Hz)
        est_damps: Estimated damping ratios
        freq_tol: Frequency tolerance (Hz)
        damp_tol: Damping tolerance

    Returns:
        results: Dictionary with comparison metrics
    """
    n_true = len(true_freqs)
    matched = np.zeros(n_true, dtype=bool)
    freq_errors = []
    damp_errors = []
    matched_pairs = []

    # For each true mode, find closest estimated mode
    for i in range(n_true):
        f_true = true_freqs[i]
        d_true = true_damps[i]

        # Find closest estimated mode
        freq_diffs = np.abs(est_freqs - f_true)
        closest_idx = np.argmin(freq_diffs)

        f_est = est_freqs[closest_idx]
        d_est = est_damps[closest_idx]

        freq_err = abs(f_est - f_true)
        damp_err = abs(d_est - d_true)

        # Check if within tolerance
        if freq_err <= freq_tol and damp_err <= damp_tol:
            matched[i] = True
            freq_errors.append(freq_err)
            damp_errors.append(damp_err)
            matched_pairs.append((i, closest_idx, freq_err, damp_err))

    n_matched = np.sum(matched)
    match_rate = n_matched / n_true if n_true > 0 else 0.0

    results = {
        'n_true': n_true,
        'n_estimated': len(est_freqs),
        'n_matched': n_matched,
        'match_rate': match_rate,
        'freq_errors': np.array(freq_errors),
        'damp_errors': np.array(damp_errors),
        'matched_pairs': matched_pairs,
        'unmatched_true': np.where(~matched)[0].tolist()
    }

    return results


def self_test_esprit_core(
    noise_level: float = 0.0,
    use_stabilization: bool = False,
    use_tls: bool = True,
    visualize: bool = True
):
    """
    Self-test for esprit_core using synthetic data.

    Args:
        noise_level: Noise standard deviation (0 = noiseless)
        use_stabilization: Use stabilization grid
        use_tls: Use TLS-ESPRIT (vs LS-ESPRIT)
        visualize: Generate visualization plots
    """
    print("="*70)
    print("ESPRIT_CORE SELF-TEST WITH SYNTHETIC DATA")
    print("="*70)

    # ========================================================================
    # 1. DEFINE GROUND TRUTH MODAL PARAMETERS
    # ========================================================================

    # Based on esprit.py self_test parameters
    fs = 787.815125  # Sampling frequency (Hz)
    dt = 1.0 / fs
    N = 260  # Number of samples

    # Ground truth modes (matching esprit.py self_test)
    true_frequencies = np.array([120, 145, 168, 185, 210, 235])  # Hz
    Q_factors = np.array([12, 20, 15, 8, 25, 18])
    true_damping_ratios = 1.0 / (2 * Q_factors)  # ζ = 1/(2Q)
    true_amplitudes = np.array([1.0, 0.9, 0.8, 0.7, 0.65, 0.6])
    true_phases = np.array([0.2, -0.6, 0.9, -1.1, 0.7, -0.3])

    n_modes = len(true_frequencies)

    print(f"\nGround Truth Parameters:")
    print(f"  Sampling frequency: {fs:.2f} Hz")
    print(f"  Signal length: {N} samples ({N/fs:.3f} s)")
    print(f"  Number of modes: {n_modes}")
    print("\n  Mode parameters:")
    print(f"  {'#':<3} {'Freq (Hz)':<10} {'Q':<8} {'Damping':<10} {'Amplitude':<10}")
    print("-" * 50)
    for i in range(n_modes):
        print(f"  {i:<3} {true_frequencies[i]:<10.1f} {Q_factors[i]:<8.1f} "
              f"{true_damping_ratios[i]:<10.4f} {true_amplitudes[i]:<10.2f}")

    # ========================================================================
    # 2. GENERATE SYNTHETIC SIGNAL
    # ========================================================================

    print(f"\nGenerating synthetic signal (noise level: {noise_level:.6f})...")

    # Single-channel signal
    signal = generate_synthetic_signal(
        n_samples=N,
        fs=fs,
        frequencies=true_frequencies,
        damping_ratios=true_damping_ratios,
        amplitudes=true_amplitudes,
        phases=true_phases,
        n_channels=1,
        noise_level=noise_level
    )

    # ========================================================================
    # 3. RUN ESPRIT_CORE MODAL IDENTIFICATION
    # ========================================================================

    # ESPRIT parameters
    window_length = 140  # L (matching esprit.py)
    model_order = 12  # K (matching esprit.py)

    print(f"\nRunning ESPRIT modal identification...")
    print(f"  Window length (L): {window_length}")
    print(f"  Model order (K): {model_order}")
    print(f"  Use TLS: {use_tls}")
    print(f"  Use stabilization: {use_stabilization}")

    result = esprit_core.esprit_modal_identification(
        signals=signal,
        fs=fs,
        window_length=window_length,
        model_order=model_order,
        use_stabilization=use_stabilization,
        use_tls=use_tls,
        use_gpu=False,
        max_damping=0.5,  # Allow higher damping for low-Q modes
        freq_range=(0, fs/2)
    )

    est_frequencies = result.frequencies
    est_damping_ratios = result.damping_ratios

    print(f"\nIdentified {len(est_frequencies)} modes")

    # ========================================================================
    # 4. COMPARE RESULTS TO GROUND TRUTH
    # ========================================================================

    print("\n" + "="*70)
    print("COMPARISON: ESTIMATED vs GROUND TRUTH")
    print("="*70)

    comparison = compare_modes(
        true_freqs=true_frequencies,
        true_damps=true_damping_ratios,
        est_freqs=est_frequencies,
        est_damps=est_damping_ratios,
        freq_tol=2.0,  # ±2 Hz tolerance
        damp_tol=0.02  # ±0.02 damping tolerance
    )

    print(f"\nSummary:")
    print(f"  True modes: {comparison['n_true']}")
    print(f"  Estimated modes: {comparison['n_estimated']}")
    print(f"  Matched modes: {comparison['n_matched']}")
    print(f"  Match rate: {comparison['match_rate']*100:.1f}%")

    if comparison['n_matched'] > 0:
        print(f"\n  Average errors (for matched modes):")
        print(f"    Frequency error: {np.mean(comparison['freq_errors']):.3f} Hz "
              f"(std: {np.std(comparison['freq_errors']):.3f} Hz)")
        print(f"    Damping error: {np.mean(comparison['damp_errors']):.4f} "
              f"(std: {np.std(comparison['damp_errors']):.4f})")

        print(f"\n  Matched pairs (true_idx, est_idx, freq_err, damp_err):")
        for true_idx, est_idx, freq_err, damp_err in comparison['matched_pairs']:
            print(f"    Mode {true_idx}: f_true={true_frequencies[true_idx]:.1f} Hz, "
                  f"f_est={est_frequencies[est_idx]:.1f} Hz, "
                  f"df={freq_err:.2f} Hz, dz={damp_err:.4f}")

    if comparison['unmatched_true']:
        print(f"\n  Unmatched true modes: {comparison['unmatched_true']}")
        for idx in comparison['unmatched_true']:
            print(f"    Mode {idx}: f={true_frequencies[idx]:.1f} Hz, "
                  f"zeta={true_damping_ratios[idx]:.4f}")

    # ========================================================================
    # 5. VALIDATE RESULTS
    # ========================================================================

    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    # Success criteria
    min_match_rate = 0.8  # At least 80% of modes should be matched
    max_freq_error = 1.0  # Maximum average frequency error (Hz)
    max_damp_error = 0.01  # Maximum average damping error

    success = True

    if comparison['match_rate'] < min_match_rate:
        print(f"[FAIL] Match rate {comparison['match_rate']*100:.1f}% < {min_match_rate*100:.1f}%")
        success = False
    else:
        print(f"[PASS] Match rate {comparison['match_rate']*100:.1f}% >= {min_match_rate*100:.1f}%")

    if comparison['n_matched'] > 0:
        avg_freq_err = np.mean(comparison['freq_errors'])
        avg_damp_err = np.mean(comparison['damp_errors'])

        if avg_freq_err > max_freq_error:
            print(f"[FAIL] Avg frequency error {avg_freq_err:.3f} Hz > {max_freq_error} Hz")
            success = False
        else:
            print(f"[PASS] Avg frequency error {avg_freq_err:.3f} Hz <= {max_freq_error} Hz")

        if avg_damp_err > max_damp_error:
            print(f"[FAIL] Avg damping error {avg_damp_err:.4f} > {max_damp_error}")
            success = False
        else:
            print(f"[PASS] Avg damping error {avg_damp_err:.4f} <= {max_damp_error}")

    if success:
        print("\n*** SELF-TEST PASSED! ***")
    else:
        print("\n*** SELF-TEST FAILED! ***")

    # ========================================================================
    # 6. VISUALIZATION
    # ========================================================================

    if visualize:
        visualize_self_test(
            signal=signal,
            fs=fs,
            true_frequencies=true_frequencies,
            true_damping_ratios=true_damping_ratios,
            true_amplitudes=true_amplitudes,
            est_frequencies=est_frequencies,
            est_damping_ratios=est_damping_ratios,
            comparison=comparison,
            noise_level=noise_level,
            use_stabilization=use_stabilization,
            use_tls=use_tls
        )

    return success, comparison


def visualize_self_test(
    signal: np.ndarray,
    fs: float,
    true_frequencies: np.ndarray,
    true_damping_ratios: np.ndarray,
    true_amplitudes: np.ndarray,
    est_frequencies: np.ndarray,
    est_damping_ratios: np.ndarray,
    comparison: dict,
    noise_level: float,
    use_stabilization: bool,
    use_tls: bool
):
    """Visualize self-test results."""

    # Sort modes by frequency for consistent numbering
    true_sort_idx = np.argsort(true_frequencies)
    true_frequencies = true_frequencies[true_sort_idx]
    true_damping_ratios = true_damping_ratios[true_sort_idx]
    true_amplitudes = true_amplitudes[true_sort_idx]

    est_sort_idx = np.argsort(est_frequencies)
    est_frequencies = est_frequencies[est_sort_idx]
    est_damping_ratios = est_damping_ratios[est_sort_idx]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    t = np.arange(len(signal)) / fs

    # Time series
    ax_time = fig.add_subplot(gs[0, :2])
    ax_time.plot(t, signal[:, 0], 'b-', linewidth=0.8, alpha=0.7, label='Synthetic signal')
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Amplitude')
    ax_time.set_title(f'Synthetic Multi-Mode Signal (noise={noise_level:.1e})')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    # FFT
    ax_fft = fig.add_subplot(gs[0, 2])
    fft_vals = np.fft.rfft(signal[:, 0])
    fft_freqs = np.fft.rfftfreq(len(signal), 1/fs)
    ax_fft.semilogy(fft_freqs, np.abs(fft_vals), 'b-', linewidth=0.5)

    # Mark true frequencies
    for f in true_frequencies:
        ax_fft.axvline(f, color='g', linestyle='--', alpha=0.5, linewidth=1)

    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Magnitude')
    ax_fft.set_title('Frequency Spectrum')
    ax_fft.grid(True, alpha=0.3)
    ax_fft.set_xlim([0, fs/2])

    # Frequency comparison
    ax_freq = fig.add_subplot(gs[1, 0])
    x_true = np.arange(len(true_frequencies))
    x_est = np.arange(len(est_frequencies))

    ax_freq.scatter(x_true, true_frequencies, s=100, marker='o',
                   color='green', label='True', alpha=0.7, edgecolors='black', linewidth=2)
    ax_freq.scatter(x_est, est_frequencies, s=80, marker='x',
                   color='red', label='Estimated', alpha=0.9, linewidth=2)

    # Draw lines connecting matched pairs
    for true_idx, est_idx, _, _ in comparison['matched_pairs']:
        ax_freq.plot([true_idx, est_idx],
                    [true_frequencies[true_idx], est_frequencies[est_idx]],
                    'k--', alpha=0.3, linewidth=1)

    ax_freq.set_xlabel('Mode Index')
    ax_freq.set_ylabel('Frequency (Hz)')
    ax_freq.set_title('Frequency Comparison')
    ax_freq.legend()
    ax_freq.grid(True, alpha=0.3)

    # Damping comparison
    ax_damp = fig.add_subplot(gs[1, 1])
    ax_damp.scatter(x_true, true_damping_ratios * 100, s=100, marker='o',
                   color='green', label='True', alpha=0.7, edgecolors='black', linewidth=2)
    ax_damp.scatter(x_est, est_damping_ratios * 100, s=80, marker='x',
                   color='red', label='Estimated', alpha=0.9, linewidth=2)

    # Draw lines connecting matched pairs
    for true_idx, est_idx, _, _ in comparison['matched_pairs']:
        ax_damp.plot([true_idx, est_idx],
                    [true_damping_ratios[true_idx] * 100, est_damping_ratios[est_idx] * 100],
                    'k--', alpha=0.3, linewidth=1)

    ax_damp.set_xlabel('Mode Index')
    ax_damp.set_ylabel('Damping Ratio (%)')
    ax_damp.set_title('Damping Comparison')
    ax_damp.legend()
    ax_damp.grid(True, alpha=0.3)

    # Frequency vs Damping scatter
    ax_scatter = fig.add_subplot(gs[1, 2])
    ax_scatter.scatter(true_frequencies, true_damping_ratios * 100,
                      s=100, marker='o', color='green', label='True',
                      alpha=0.7, edgecolors='black', linewidth=2)
    ax_scatter.scatter(est_frequencies, est_damping_ratios * 100,
                      s=80, marker='x', color='red', label='Estimated',
                      alpha=0.9, linewidth=2)

    ax_scatter.set_xlabel('Frequency (Hz)')
    ax_scatter.set_ylabel('Damping Ratio (%)')
    ax_scatter.set_title('Frequency vs Damping')
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)

    # Overall title
    method_str = "TLS-ESPRIT" if use_tls else "LS-ESPRIT"
    stab_str = " + Stabilization" if use_stabilization else ""
    match_str = f"{comparison['n_matched']}/{comparison['n_true']} matched ({comparison['match_rate']*100:.0f}%)"

    fig.suptitle(f'ESPRIT_CORE Self-Test: {method_str}{stab_str} | {match_str}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def run_noise_sensitivity_test():
    """Test ESPRIT performance across different noise levels."""
    print("\n" + "="*70)
    print("NOISE SENSITIVITY TEST")
    print("="*70)

    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    results = []

    for noise in noise_levels:
        print(f"\n--- Testing noise level: {noise:.4f} ---")
        success, comparison = self_test_esprit_core(
            noise_level=noise,
            use_stabilization=False,
            use_tls=True,
            visualize=False
        )
        results.append({
            'noise': noise,
            'match_rate': comparison['match_rate'],
            'n_matched': comparison['n_matched'],
            'avg_freq_err': np.mean(comparison['freq_errors']) if comparison['n_matched'] > 0 else np.nan,
            'avg_damp_err': np.mean(comparison['damp_errors']) if comparison['n_matched'] > 0 else np.nan
        })

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    noise_vals = [r['noise'] for r in results]
    match_rates = [r['match_rate'] * 100 for r in results]
    freq_errs = [r['avg_freq_err'] for r in results]
    damp_errs = [r['avg_damp_err'] * 100 for r in results]  # Convert to %

    axes[0].plot(noise_vals, match_rates, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Noise Level (std)')
    axes[0].set_ylabel('Match Rate (%)')
    axes[0].set_title('Match Rate vs Noise')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 105])

    axes[1].plot(noise_vals, freq_errs, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Noise Level (std)')
    axes[1].set_ylabel('Avg Frequency Error (Hz)')
    axes[1].set_title('Frequency Accuracy vs Noise')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(noise_vals, damp_errs, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Noise Level (std)')
    axes[2].set_ylabel('Avg Damping Error (%)')
    axes[2].set_title('Damping Accuracy vs Noise')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('ESPRIT Noise Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.show()


if __name__ == "__main__":
    # Run self-test with different configurations

    print("\n" + "#"*70)
    print("# TEST 1: TLS-ESPRIT (no noise, no stabilization)")
    print("#"*70)
    self_test_esprit_core(
        noise_level=0.0,
        use_stabilization=False,
        use_tls=True,
        visualize=True
    )

    print("\n" + "#"*70)
    print("# TEST 2: LS-ESPRIT (no noise, no stabilization)")
    print("#"*70)
    self_test_esprit_core(
        noise_level=0.0,
        use_stabilization=False,
        use_tls=False,
        visualize=True
    )

    print("\n" + "#"*70)
    print("# TEST 3: TLS-ESPRIT + Stabilization (no noise)")
    print("#"*70)
    self_test_esprit_core(
        noise_level=0.0,
        use_stabilization=True,
        use_tls=True,
        visualize=True
    )

    print("\n" + "#"*70)
    print("# TEST 4: TLS-ESPRIT with noise (SNR ~30dB)")
    print("#"*70)
    self_test_esprit_core(
        noise_level=0.01,
        use_stabilization=False,
        use_tls=True,
        visualize=True
    )

    # Run noise sensitivity analysis
    run_noise_sensitivity_test()
