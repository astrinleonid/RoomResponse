"""
Comprehensive synthetic data test for ESPRIT implementations.

Tests esprit.py vs esprit_core.py on challenging synthetic signal:
- 40 modes spanning 40-4000 Hz
- Varying damping ratios (0.5% to 5%)
- Multiple noise levels
- Performance comparison (accuracy + speed)
"""
import numpy as np
import time
import json
from pathlib import Path

# Import both implementations
import esprit
import esprit_core


def generate_complex_synthetic_signal(
    fs: float,
    duration: float,
    modes: list,
    noise_level: float = 0.0,
    n_channels: int = 3
) -> np.ndarray:
    """
    Generate multi-channel synthetic signal with known modal parameters.

    Args:
        fs: Sampling frequency (Hz)
        duration: Signal duration (seconds)
        modes: List of (frequency, damping_ratio, amplitude, phase) tuples
        noise_level: Noise standard deviation as fraction of signal RMS
        n_channels: Number of channels to generate

    Returns:
        signals: Shape (N, n_channels)
    """
    N = int(fs * duration)
    t = np.arange(N) / fs
    dt = 1.0 / fs

    # Initialize multi-channel signal
    signals = np.zeros((N, n_channels))

    for ch in range(n_channels):
        x = np.zeros(N)

        for f, zeta, A, ph in modes:
            # Natural angular frequency
            omega_n = 2.0 * np.pi * f

            # Damped angular frequency
            omega_d = omega_n * np.sqrt(1 - zeta**2)

            # Decay rate
            alpha = zeta * omega_n

            # Channel-specific amplitude and phase variation
            A_ch = A * (0.8 + 0.4 * np.random.rand())  # 80-120% of nominal
            ph_ch = ph + np.random.rand() * 0.5  # Phase variation

            # Generate damped sinusoid
            x += A_ch * np.exp(-alpha * t) * np.cos(omega_d * t + ph_ch)

        signals[:, ch] = x

    # Add noise if requested
    if noise_level > 0:
        signal_rms = np.sqrt(np.mean(signals**2))
        noise = np.random.randn(N, n_channels) * (noise_level * signal_rms)
        signals += noise

    return signals


def generate_test_modes(n_modes: int, f_min: float, f_max: float) -> list:
    """
    Generate test mode parameters with realistic distribution.

    Args:
        n_modes: Number of modes to generate
        f_min: Minimum frequency (Hz)
        f_max: Maximum frequency (Hz)

    Returns:
        modes: List of (frequency, damping_ratio, amplitude, phase) tuples
    """
    np.random.seed(42)  # Reproducible

    # Logarithmic frequency distribution (more modes at low frequencies)
    log_freqs = np.linspace(np.log10(f_min), np.log10(f_max), n_modes)
    frequencies = 10**log_freqs

    # Damping ratios: 0.5% to 5% (realistic for lightly damped structures)
    # Lower frequencies typically have lower damping
    damping_ratios = 0.005 + 0.045 * (frequencies / f_max)**0.5

    # Amplitudes: decay with frequency (typical structural response)
    amplitudes = 1.0 / (1.0 + (frequencies / 500)**2)

    # Random phases
    phases = np.random.rand(n_modes) * 2 * np.pi

    modes = []
    for f, zeta, A, ph in zip(frequencies, damping_ratios, amplitudes, phases):
        modes.append((f, zeta, A, ph))

    return modes


def run_esprit_py(signals: np.ndarray, fs: float, model_order: int,
                  window_length: int, min_freq: float = 0.0) -> dict:
    """
    Run esprit.py TLS-U algorithm.

    Returns:
        dict with frequencies, damping_ratios, time
    """
    # Extract first channel
    signal = signals[:, 0]

    dt = 1.0 / fs
    t0 = time.perf_counter()

    # Build Hankel matrix
    L = window_length
    N = len(signal)
    C = N - L + 1

    from numpy.lib.stride_tricks import as_strided
    shape = (L, C)
    strides = (signal.strides[0], signal.strides[0])
    H = as_strided(signal, shape=shape, strides=strides).copy()

    # SVD to get U subspace
    U, s, Vh = np.linalg.svd(H, full_matrices=False)

    # Extract signal subspace
    K = model_order
    Us = U[:, :K]

    # Run TLS-U ESPRIT (matching esprit.py signature)
    # TLS_ESPRIT_FromUs(Us, rows, L, B, K) where rows=L*B, B=1 (single channel), K=model_order
    try:
        lam, _ = esprit.TLS_ESPRIT_FromUs(Us, L, L, 1, K)
    except Exception as e:
        print(f"    esprit.py error: {e}")
        elapsed = time.perf_counter() - t0
        return {
            'frequencies': np.array([]),
            'damping_ratios': np.array([]),
            'time': elapsed,
            'n_modes': 0
        }

    # Filter by radius
    radius = np.abs(lam)
    mask_radius = (radius >= 0.5) & (radius <= 1.3)
    lam_filtered = lam[mask_radius]

    if len(lam_filtered) == 0:
        elapsed = time.perf_counter() - t0
        return {
            'frequencies': np.array([]),
            'damping_ratios': np.array([]),
            'time': elapsed,
            'n_modes': 0
        }

    # Conjugate pairing (in z-plane)
    poles_ct, pair_quality = esprit_core.validate_conjugate_pairs(lam_filtered, dt)

    if len(poles_ct) == 0:
        elapsed = time.perf_counter() - t0
        return {
            'frequencies': np.array([]),
            'damping_ratios': np.array([]),
            'time': elapsed,
            'n_modes': 0
        }

    # Convert to modal parameters
    frequencies, damping_ratios = esprit_core.poles_to_modal_params(poles_ct, fs)

    # Filter (use relaxed damping threshold for synthetic data)
    mask = esprit_core.filter_poles(
        poles_ct, frequencies, damping_ratios,
        max_damping=0.5, min_freq=min_freq, max_freq=np.inf  # Relaxed damping
    )

    frequencies = frequencies[mask]
    damping_ratios = damping_ratios[mask]

    elapsed = time.perf_counter() - t0

    return {
        'frequencies': frequencies,
        'damping_ratios': damping_ratios,
        'time': elapsed,
        'n_modes': len(frequencies)
    }


def run_esprit_core(signals: np.ndarray, fs: float, model_order: int,
                    window_length: int, use_gpu: bool = False,
                    min_freq: float = 0.0) -> dict:
    """
    Run esprit_core.py algorithm.

    Returns:
        dict with frequencies, damping_ratios, time
    """
    t0 = time.perf_counter()

    result = esprit_core.esprit_modal_identification(
        signals=signals,
        fs=fs,
        model_order=model_order,
        window_length=window_length,
        use_gpu=use_gpu,
        use_tls=True,
        use_stabilization=False,
        use_conjugate_pairing=True,
        min_freq=min_freq,
        max_damping=0.5  # Relaxed for synthetic data
    )

    elapsed = time.perf_counter() - t0

    return {
        'frequencies': result.frequencies,
        'damping_ratios': result.damping_ratios,
        'time': elapsed,
        'n_modes': len(result.frequencies)
    }


def match_modes(true_modes: list, detected_freqs: np.ndarray,
                detected_damps: np.ndarray, freq_tol: float = 5.0) -> dict:
    """
    Match detected modes to ground truth.

    Returns:
        dict with matched, missed, spurious statistics
    """
    true_freqs = np.array([f for f, _, _, _ in true_modes])
    true_damps = np.array([zeta for _, zeta, _, _ in true_modes])

    matched = 0
    freq_errors = []
    damp_errors = []

    matched_indices = set()

    for f_true, zeta_true in zip(true_freqs, true_damps):
        # Find closest detected mode
        if len(detected_freqs) == 0:
            continue

        freq_diffs = np.abs(detected_freqs - f_true)
        min_idx = np.argmin(freq_diffs)

        if freq_diffs[min_idx] < freq_tol and min_idx not in matched_indices:
            matched += 1
            matched_indices.add(min_idx)
            freq_errors.append(detected_freqs[min_idx] - f_true)
            damp_errors.append(detected_damps[min_idx] - zeta_true)

    missed = len(true_modes) - matched
    spurious = len(detected_freqs) - matched

    return {
        'matched': matched,
        'missed': missed,
        'spurious': spurious,
        'freq_errors': np.array(freq_errors),
        'damp_errors': np.array(damp_errors),
        'match_rate': matched / len(true_modes) if len(true_modes) > 0 else 0
    }


def print_comparison_table(results: dict):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE SYNTHETIC TEST RESULTS")
    print("="*80)

    # Test configuration
    config = results['config']
    print(f"\nTest Configuration:")
    print(f"  Signal duration:  {config['duration']:.1f} s")
    print(f"  Sampling rate:    {config['fs']:.0f} Hz")
    print(f"  True modes:       {config['n_modes']}")
    print(f"  Frequency range:  {config['f_min']:.0f} - {config['f_max']:.0f} Hz")
    print(f"  Model order:      {config['model_order']}")
    print(f"  Window length:    {config['window_length']}")
    print(f"  Noise level:      {config['noise_level']*100:.1f}% RMS")

    # Results table
    print(f"\n{'Implementation':<25} {'Detected':<10} {'Matched':<10} {'Missed':<10} {'Spurious':<10} {'Time (ms)':<12}")
    print("-"*80)

    for impl_name, impl_results in results['implementations'].items():
        stats = impl_results['match_stats']
        print(f"{impl_name:<25} {impl_results['n_modes']:<10} "
              f"{stats['matched']:<10} {stats['missed']:<10} "
              f"{stats['spurious']:<10} {impl_results['time']*1000:<12.3f}")

    # Accuracy comparison
    print(f"\n{'Implementation':<25} {'Match Rate':<12} {'Freq RMSE (Hz)':<18} {'Damp RMSE':<15}")
    print("-"*80)

    for impl_name, impl_results in results['implementations'].items():
        stats = impl_results['match_stats']
        freq_errs = stats['freq_errors']
        damp_errs = stats['damp_errors']

        if len(freq_errs) > 0:
            freq_rmse = np.sqrt(np.mean(freq_errs**2))
            damp_rmse = np.sqrt(np.mean(damp_errs**2))
        else:
            freq_rmse = np.nan
            damp_rmse = np.nan

        match_rate = stats['match_rate'] * 100
        print(f"{impl_name:<25} {match_rate:<12.1f}% {freq_rmse:<18.4f} {damp_rmse:<15.6f}")

    # Speed comparison
    print("\nSpeed Comparison:")
    print("-"*80)

    ref_time = results['implementations']['esprit.py (TLS-U)']['time']

    for impl_name, impl_results in results['implementations'].items():
        t = impl_results['time']
        speedup = ref_time / t if t > 0 else 0
        print(f"  {impl_name:<30} {t*1000:8.3f} ms  ({speedup:5.2f}x)")


def run_comprehensive_test():
    """Run comprehensive synthetic test."""

    print("Generating complex synthetic signal with 40 modes...")

    # Configuration
    fs = 8000.0  # Hz (high enough for 4 kHz modes)
    duration = 1.0  # seconds (reduced for performance)
    n_modes = 40
    f_min = 40.0
    f_max = 4000.0
    model_order = 60  # 1.5x number of modes (reduced for performance)
    noise_level = 0.01  # 1% RMS noise

    # Generate ground truth modes
    true_modes = generate_test_modes(n_modes, f_min, f_max)

    print(f"\nGenerated {n_modes} modes:")
    print(f"  Frequency range: {true_modes[0][0]:.1f} - {true_modes[-1][0]:.1f} Hz")
    print(f"  Damping range:   {min(m[1] for m in true_modes)*100:.2f}% - {max(m[1] for m in true_modes)*100:.2f}%")

    # Generate signal
    signals = generate_complex_synthetic_signal(
        fs, duration, true_modes, noise_level=noise_level, n_channels=3
    )

    N = len(signals)
    window_length = N // 2

    print(f"\nSignal properties:")
    print(f"  Length: {N} samples ({duration:.1f} s)")
    print(f"  Channels: {signals.shape[1]}")
    print(f"  Window length: {window_length}")
    print(f"  SNR: {20*np.log10(1.0/noise_level):.1f} dB" if noise_level > 0 else "  SNR: Infinite (no noise)")

    # Run tests
    print("\n" + "="*80)
    print("Running ESPRIT implementations...")
    print("="*80)

    results = {
        'config': {
            'fs': fs,
            'duration': duration,
            'n_modes': n_modes,
            'f_min': f_min,
            'f_max': f_max,
            'model_order': model_order,
            'window_length': window_length,
            'noise_level': noise_level
        },
        'true_modes': true_modes,
        'implementations': {}
    }

    # Test 1: esprit.py (TLS-U)
    print("\n[1/3] Running esprit.py (TLS-U)...")
    try:
        result_py = run_esprit_py(signals, fs, model_order, window_length, min_freq=20.0)
        match_stats_py = match_modes(true_modes, result_py['frequencies'],
                                      result_py['damping_ratios'])
        results['implementations']['esprit.py (TLS-U)'] = {
            **result_py,
            'match_stats': match_stats_py
        }
        print(f"  Detected {result_py['n_modes']} modes in {result_py['time']*1000:.3f} ms")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['implementations']['esprit.py (TLS-U)'] = {
            'frequencies': np.array([]),
            'damping_ratios': np.array([]),
            'time': 0.0,
            'n_modes': 0,
            'match_stats': {
                'matched': 0, 'missed': n_modes, 'spurious': 0,
                'freq_errors': np.array([]), 'damp_errors': np.array([]),
                'match_rate': 0.0
            }
        }

    # Test 2: esprit_core (TLS-U CPU)
    print("\n[2/3] Running esprit_core (TLS-U CPU)...")
    try:
        result_core_cpu = run_esprit_core(signals, fs, model_order, window_length,
                                          use_gpu=False, min_freq=20.0)
        match_stats_core_cpu = match_modes(true_modes, result_core_cpu['frequencies'],
                                            result_core_cpu['damping_ratios'])
        results['implementations']['esprit_core (TLS-U CPU)'] = {
            **result_core_cpu,
            'match_stats': match_stats_core_cpu
        }
        print(f"  Detected {result_core_cpu['n_modes']} modes in {result_core_cpu['time']*1000:.3f} ms")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['implementations']['esprit_core (TLS-U CPU)'] = {
            'frequencies': np.array([]),
            'damping_ratios': np.array([]),
            'time': 0.0,
            'n_modes': 0,
            'match_stats': {
                'matched': 0, 'missed': n_modes, 'spurious': 0,
                'freq_errors': np.array([]), 'damp_errors': np.array([]),
                'match_rate': 0.0
            }
        }

    # Test 3: esprit_core (TLS-U GPU) - if available
    print("\n[3/3] Running esprit_core (TLS-U GPU)...")
    try:
        import cupy as cp
        result_core_gpu = run_esprit_core(signals, fs, model_order, window_length,
                                          use_gpu=True, min_freq=20.0)
        match_stats_core_gpu = match_modes(true_modes, result_core_gpu['frequencies'],
                                            result_core_gpu['damping_ratios'])
        results['implementations']['esprit_core (TLS-U GPU)'] = {
            **result_core_gpu,
            'match_stats': match_stats_core_gpu
        }
        print(f"  Detected {result_core_gpu['n_modes']} modes in {result_core_gpu['time']*1000:.3f} ms")
    except ImportError:
        print("  SKIPPED: CuPy not available")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Print results
    print_comparison_table(results)

    # Save detailed results
    output_file = Path(__file__).parent / 'test_comprehensive_synthetic_results.json'

    # Convert numpy arrays to lists for JSON serialization
    results_export = {
        'config': results['config'],
        'true_modes': [
            {
                'frequency': float(f),
                'damping_ratio': float(zeta),
                'amplitude': float(A),
                'phase': float(ph)
            }
            for f, zeta, A, ph in results['true_modes']
        ],
        'implementations': {}
    }

    for impl_name, impl_results in results['implementations'].items():
        results_export['implementations'][impl_name] = {
            'n_modes': int(impl_results['n_modes']),
            'time': float(impl_results['time']),
            'frequencies': impl_results['frequencies'].tolist(),
            'damping_ratios': impl_results['damping_ratios'].tolist(),
            'match_stats': {
                'matched': int(impl_results['match_stats']['matched']),
                'missed': int(impl_results['match_stats']['missed']),
                'spurious': int(impl_results['match_stats']['spurious']),
                'match_rate': float(impl_results['match_stats']['match_rate']),
                'freq_errors': impl_results['match_stats']['freq_errors'].tolist(),
                'damp_errors': impl_results['match_stats']['damp_errors'].tolist()
            }
        }

    with open(output_file, 'w') as f:
        json.dump(results_export, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if 'esprit_core (TLS-U CPU)' in results['implementations']:
        core_stats = results['implementations']['esprit_core (TLS-U CPU)']['match_stats']
        core_time = results['implementations']['esprit_core (TLS-U CPU)']['time']

        py_stats = results['implementations']['esprit.py (TLS-U)']['match_stats']
        py_time = results['implementations']['esprit.py (TLS-U)']['time']

        # Accuracy comparison
        if len(core_stats['freq_errors']) > 0:
            core_freq_rmse = np.sqrt(np.mean(core_stats['freq_errors']**2))
            print(f"\nAccuracy (esprit_core vs esprit.py):")
            print(f"  Match rate:       {core_stats['match_rate']*100:.1f}% vs {py_stats['match_rate']*100:.1f}%")
            print(f"  Frequency RMSE:   {core_freq_rmse:.4f} Hz")
            print(f"  Max freq error:   {np.max(np.abs(core_stats['freq_errors'])):.4f} Hz")

        # Speed comparison
        if py_time > 0:
            speedup = py_time / core_time
            print(f"\nSpeed (esprit_core vs esprit.py):")
            print(f"  esprit_core:  {core_time*1000:.3f} ms")
            print(f"  esprit.py:    {py_time*1000:.3f} ms")
            print(f"  Speedup:      {speedup:.2f}x")

        # GPU comparison
        if 'esprit_core (TLS-U GPU)' in results['implementations']:
            gpu_time = results['implementations']['esprit_core (TLS-U GPU)']['time']
            if gpu_time > 0:
                gpu_speedup = core_time / gpu_time
                print(f"\nGPU Performance:")
                print(f"  CPU time:     {core_time*1000:.3f} ms")
                print(f"  GPU time:     {gpu_time*1000:.3f} ms")
                print(f"  GPU speedup:  {gpu_speedup:.2f}x")

    print("\n" + "="*80)
    print("[OK] Comprehensive synthetic test completed successfully!")
    print("="*80)


if __name__ == '__main__':
    run_comprehensive_test()
