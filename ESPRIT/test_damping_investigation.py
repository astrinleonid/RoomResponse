"""
Investigation: Why is damping estimation failing above 1kHz?

This test explores:
1. Effect of signal duration on damping estimation
2. Effect of amplitude/SNR on damping estimation
3. Effect of window length on damping estimation
4. Comparison of TLS vs LS for damping estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import esprit_core

def generate_single_mode_signal(fs, duration, freq, damping, amplitude, noise_level):
    """Generate a signal with a single damped mode."""
    N = int(fs * duration)
    t = np.arange(N) / fs

    omega_n = 2 * np.pi * freq
    alpha = damping * omega_n
    omega_d = omega_n * np.sqrt(1 - damping**2)

    x = amplitude * np.exp(-alpha * t) * np.cos(omega_d * t)

    if noise_level > 0:
        noise_std = noise_level * np.sqrt(np.mean(x**2))
        x += np.random.normal(0, noise_std, N)

    return x.reshape(-1, 1), t


def test_damping_accuracy(freq, true_damping, amplitude, fs=8000, durations=[1, 2, 4, 8],
                          noise_levels=[0.0, 0.01, 0.05], use_tls_values=[True, False]):
    """Test damping estimation accuracy for different conditions."""

    results = []

    for duration in durations:
        for noise in noise_levels:
            for use_tls in use_tls_values:

                np.random.seed(42)  # Reproducibility
                signals, t = generate_single_mode_signal(
                    fs, duration, freq, true_damping, amplitude, noise
                )

                # ESPRIT parameters
                model_order = 4  # 2 modes (real + conjugate)
                window_length = min(int(len(signals) * 0.8), len(signals) - model_order)

                try:
                    result = esprit_core.esprit_modal_identification(
                        signals,
                        fs=fs,
                        model_order=model_order,
                        window_length=window_length,
                        use_gpu=False,
                        use_tls=use_tls,
                        use_stabilization=False,
                        use_conjugate_pairing=True,
                        min_freq=freq * 0.8,
                        freq_range=(freq * 0.8, freq * 1.2),
                        max_damping=0.2
                    )

                    if len(result.frequencies) > 0:
                        # Find closest mode
                        idx = np.argmin(np.abs(result.frequencies - freq))
                        detected_freq = result.frequencies[idx]
                        detected_damp = result.damping_ratios[idx]

                        freq_error = detected_freq - freq
                        damp_error = detected_damp - true_damping
                        damp_rel_error = (detected_damp - true_damping) / true_damping * 100

                        results.append({
                            'freq': freq,
                            'duration': duration,
                            'noise': noise,
                            'use_tls': use_tls,
                            'true_damping': true_damping,
                            'detected_freq': detected_freq,
                            'detected_damp': detected_damp,
                            'freq_error': freq_error,
                            'damp_error': damp_error,
                            'damp_rel_error': damp_rel_error,
                            'success': True
                        })
                    else:
                        results.append({
                            'freq': freq,
                            'duration': duration,
                            'noise': noise,
                            'use_tls': use_tls,
                            'true_damping': true_damping,
                            'success': False
                        })

                except Exception as e:
                    results.append({
                        'freq': freq,
                        'duration': duration,
                        'noise': noise,
                        'use_tls': use_tls,
                        'true_damping': true_damping,
                        'error': str(e),
                        'success': False
                    })

    return results


def main():
    """Run damping estimation investigation."""

    print("="*80)
    print("DAMPING ESTIMATION INVESTIGATION")
    print("="*80)
    print()

    # Test frequencies
    test_cases = [
        {'freq': 100, 'damping': 0.02, 'amplitude': 1.0, 'label': 'Low freq, high amp'},
        {'freq': 500, 'damping': 0.025, 'amplitude': 0.5, 'label': 'Mid freq, mid amp'},
        {'freq': 1500, 'damping': 0.034, 'amplitude': 0.2, 'label': 'High freq, low amp'},
        {'freq': 3000, 'damping': 0.038, 'amplitude': 0.15, 'label': 'Very high freq, very low amp'},
    ]

    all_results = {}

    for case in test_cases:
        print(f"\nTesting: {case['label']}")
        print(f"  Frequency: {case['freq']} Hz")
        print(f"  True damping: {case['damping']*100:.1f}%")
        print(f"  Amplitude: {case['amplitude']}")

        results = test_damping_accuracy(
            freq=case['freq'],
            true_damping=case['damping'],
            amplitude=case['amplitude'],
            durations=[2, 4, 8],  # Test different durations
            noise_levels=[0.0, 0.01],  # Clean and realistic noise
            use_tls_values=[True, False]  # TLS vs LS
        )

        all_results[case['freq']] = results

        # Print best result for this frequency
        successful = [r for r in results if r['success']]
        if successful:
            best = min(successful, key=lambda r: abs(r['damp_rel_error']))
            print(f"  Best result:")
            print(f"    Duration: {best['duration']}s, Noise: {best['noise']}, TLS: {best['use_tls']}")
            print(f"    Detected damping: {best['detected_damp']*100:.2f}% (error: {best['damp_rel_error']:+.1f}%)")

    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    # Analyze effect of duration
    print("\nEffect of Signal Duration on Damping Estimation (3000 Hz mode):")
    results_3k = all_results[3000]
    for dur in [2, 4, 8]:
        clean_tls = [r for r in results_3k if r['success'] and r['duration'] == dur and r['noise'] == 0.0 and r['use_tls']]
        if clean_tls:
            r = clean_tls[0]
            print(f"  {dur}s: detected={r['detected_damp']*100:.3f}%, error={r['damp_rel_error']:+.1f}%")

    # Analyze effect of noise
    print("\nEffect of Noise on Damping Estimation (3000 Hz mode, 4s duration, TLS):")
    for noise in [0.0, 0.01]:
        matches = [r for r in results_3k if r['success'] and r['duration'] == 4 and r['noise'] == noise and r['use_tls']]
        if matches:
            r = matches[0]
            print(f"  Noise={noise}: detected={r['detected_damp']*100:.3f}%, error={r['damp_rel_error']:+.1f}%")

    # Analyze TLS vs LS
    print("\nTLS vs LS for Damping Estimation (3000 Hz mode, 4s, no noise):")
    for use_tls in [True, False]:
        method = "TLS" if use_tls else "LS"
        matches = [r for r in results_3k if r['success'] and r['duration'] == 4 and r['noise'] == 0.0 and r['use_tls'] == use_tls]
        if matches:
            r = matches[0]
            print(f"  {method}: detected={r['detected_damp']*100:.3f}%, error={r['damp_rel_error']:+.1f}%")

    print()

if __name__ == '__main__':
    main()
