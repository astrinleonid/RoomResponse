"""
esprit_core.py
Core ESPRIT modal identification algorithm for piano soundboard analysis.

Implements the ESPRIT method for extracting modal parameters (poles, frequencies,
damping ratios, and complex mode shapes) from multi-channel impulse response data.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ModalParameters:
    """Container for identified modal parameters."""
    poles: np.ndarray  # Complex poles s_k = σ_k + jω_k, shape (K,)
    frequencies: np.ndarray  # Natural frequencies f_k in Hz, shape (K,)
    damping_ratios: np.ndarray  # Damping ratios ζ_k, shape (K,)
    mode_shapes: np.ndarray  # Complex mode shapes φ_k, shape (K, n_channels)
    model_order: int  # Model order M used
    singular_values: np.ndarray  # Singular values from SVD

    def __repr__(self):
        return (f"ModalParameters(n_modes={len(self.poles)}, "
                f"order={self.model_order}, "
                f"f_range=[{self.frequencies.min():.2f}, {self.frequencies.max():.2f}] Hz)")


def build_hankel_matrix(signal: np.ndarray, window_length: int) -> np.ndarray:
    """
    Construct a Hankel matrix from a 1D signal using efficient stride tricks.

    Args:
        signal: 1D time series, shape (N,)
        window_length: Number of rows L in Hankel matrix

    Returns:
        H: Hankel matrix, shape (L, K) where K = N - L + 1
    """
    from numpy.lib.stride_tricks import as_strided

    N = len(signal)
    L = window_length
    K = N - L + 1

    if K <= 0:
        raise ValueError(f"Signal length {N} too short for window length {L}")

    shape = (L, K)
    strides = (signal.strides[0], signal.strides[0])
    H = as_strided(signal, shape=shape, strides=strides)

    return H.copy()


def build_multichannel_hankel(signals: np.ndarray, window_length: int,
                               mode: str = 'stack') -> np.ndarray:
    """
    Build Hankel matrix from multi-channel data.

    Args:
        signals: Multi-channel time series, shape (T, n_channels)
        window_length: Number of rows per channel
        mode: 'stack' - vertically stack Hankel matrices from each channel
              'single' - use only first channel

    Returns:
        H: Stacked Hankel matrix, shape (L*n_channels, K) for mode='stack'
           or (L, K) for mode='single'
    """
    T, n_channels = signals.shape

    if mode == 'single':
        return build_hankel_matrix(signals[:, 0], window_length)
    elif mode == 'stack':
        hankels = []
        for ch in range(n_channels):
            H_ch = build_hankel_matrix(signals[:, ch], window_length)
            hankels.append(H_ch)
        return np.vstack(hankels)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def esprit_poles(hankel_matrix: np.ndarray, model_order: int, dt: float,
                 use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract modal poles using ESPRIT algorithm.

    The ESPRIT method uses the shift structure of the Hankel matrix to identify
    the system poles without computing the full eigenvalue decomposition.

    Args:
        hankel_matrix: Hankel matrix H, shape (L, K)
        model_order: Model order M (number of poles to extract)
        dt: Time step in seconds
        use_gpu: Use GPU acceleration if available

    Returns:
        poles: Complex poles s_k, shape (M,)
        singular_values: All singular values from SVD, shape (min(L,K),)
    """
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            H = cp.asarray(hankel_matrix)
        except (ImportError, Exception) as e:
            print(f"  GPU unavailable or error ({type(e).__name__}), falling back to CPU")
            xp = np
            H = hankel_matrix
            use_gpu = False
    else:
        xp = np
        H = hankel_matrix

    # Economy SVD
    try:
        U, s, Vh = xp.linalg.svd(H, full_matrices=False)
    except Exception as e:
        if use_gpu:
            print(f"  GPU SVD failed ({type(e).__name__}), falling back to CPU")
            xp = np
            H = hankel_matrix
            use_gpu = False
            U, s, Vh = xp.linalg.svd(H, full_matrices=False)
        else:
            raise

    # Automatic subspace order selection based on energy (99% threshold)
    # This prevents overfitting to noise when model_order is too high
    energy = xp.cumsum(s**2) / xp.sum(s**2)
    M_auto = int(xp.searchsorted(energy, 0.99))
    M_use = min(model_order, max(4, M_auto))  # At least 4, at most model_order

    # Extract signal subspace (first M_use singular vectors)
    E = U[:, :M_use]

    # Split into E1 (rows 0 to L-2) and E2 (rows 1 to L-1)
    E1 = E[:-1, :]
    E2 = E[1:, :]

    # Solve for Phi: E2 = E1 * Phi
    # Use least-squares instead of pseudoinverse for numerical stability
    # lstsq is more stable than pinv and avoids noise amplification
    X, *_ = xp.linalg.lstsq(E1, E2, rcond=None)

    # Eigenvalues of Phi give the discrete-time poles
    lam = xp.linalg.eigvals(X)

    # Convert to continuous-time poles: s = ln(λ) / dt
    poles = xp.log(lam) / dt

    # Move results back to CPU if using GPU
    if use_gpu and xp.__name__ == 'cupy':
        poles = cp.asnumpy(poles)
        s = cp.asnumpy(s)

    return poles, s


def poles_to_modal_params(poles: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert complex poles to natural frequencies and damping ratios.

    For pole s_k = σ_k + jω_k:
        f_k = ω_k / (2π)
        ζ_k = -σ_k / sqrt(σ_k² + ω_k²)

    Args:
        poles: Complex poles s_k = σ_k + jω_k, shape (M,)
        fs: Sampling frequency in Hz

    Returns:
        frequencies: Natural frequencies in Hz, shape (M,)
        damping_ratios: Damping ratios (dimensionless), shape (M,)
    """
    sigma = poles.real
    omega = np.abs(poles.imag)

    frequencies = omega / (2 * np.pi)

    # Damping ratio: ζ = -σ / |s|
    pole_magnitude = np.sqrt(sigma**2 + omega**2)
    damping_ratios = -sigma / pole_magnitude

    return frequencies, damping_ratios


def estimate_mode_shapes(signals: np.ndarray, poles: np.ndarray, dt: float) -> np.ndarray:
    """
    Estimate complex mode shapes using least squares.

    For each channel i, fit the decay as:
        y_i[n] ≈ Σ_k φ_k(i) · e^{s_k n Δt}

    Args:
        signals: Multi-channel decay signals, shape (T, n_channels)
        poles: Complex poles s_k, shape (M,)
        dt: Time step in seconds

    Returns:
        mode_shapes: Complex mode shapes, shape (M, n_channels)
    """
    T, n_channels = signals.shape
    M = len(poles)

    # Build Vandermonde matrix Z: Z[n, k] = e^{s_k n Δt}
    n = np.arange(T)
    Z = np.exp(np.outer(n, poles) * dt)  # shape (T, M)

    # Solve for mode shapes using least squares for each channel
    mode_shapes = np.zeros((M, n_channels), dtype=complex)

    for ch in range(n_channels):
        # Least squares: minimize ||Z @ φ - y_ch||²
        phi_ch, _, _, _ = np.linalg.lstsq(Z, signals[:, ch], rcond=None)
        mode_shapes[:, ch] = phi_ch

    return mode_shapes


def normalize_mode_shapes(mode_shapes: np.ndarray, ref_sensor: int = 0) -> np.ndarray:
    """
    Normalize mode shapes: max magnitude = 1, reference phase = 0.

    Args:
        mode_shapes: Complex mode shapes, shape (M, n_channels)
        ref_sensor: Reference sensor index for phase normalization

    Returns:
        normalized_shapes: Normalized mode shapes
    """
    M, n_channels = mode_shapes.shape
    normalized = np.zeros_like(mode_shapes)

    for k in range(M):
        # Find max magnitude
        max_idx = np.argmax(np.abs(mode_shapes[k, :]))
        max_val = mode_shapes[k, max_idx]

        # Scale by magnitude
        shape_scaled = mode_shapes[k, :] / np.abs(max_val)

        # Rotate phase so reference sensor has phase = 0
        if ref_sensor < n_channels:
            ref_phase = np.angle(shape_scaled[ref_sensor])
            shape_normalized = shape_scaled * np.exp(-1j * ref_phase)
        else:
            shape_normalized = shape_scaled

        normalized[k, :] = shape_normalized

    return normalized


def cluster_poles(poles_list: list, frequencies_list: list, damping_list: list,
                  freq_tol_hz: float = 5.0, damping_tol: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster poles from multiple (M, L) combinations to find stable modes.

    Poles that appear consistently across different model orders and window lengths
    are likely physical modes, while spurious poles will not cluster.

    Args:
        poles_list: List of pole arrays from different (M, L) combinations
        frequencies_list: List of frequency arrays
        damping_list: List of damping ratio arrays
        freq_tol_hz: Frequency tolerance for clustering (Hz)
        damping_tol: Damping ratio tolerance for clustering

    Returns:
        clustered_poles: Poles that appear in ≥2 combinations
        clustered_frequencies: Corresponding frequencies
        clustered_damping: Corresponding damping ratios
    """
    if len(poles_list) == 0:
        return np.array([]), np.array([]), np.array([])

    # Flatten all poles
    all_poles = []
    all_freqs = []
    all_damps = []
    for poles, freqs, damps in zip(poles_list, frequencies_list, damping_list):
        all_poles.extend(poles)
        all_freqs.extend(freqs)
        all_damps.extend(damps)

    if len(all_poles) == 0:
        return np.array([]), np.array([]), np.array([])

    all_poles = np.array(all_poles)
    all_freqs = np.array(all_freqs)
    all_damps = np.array(all_damps)

    # Cluster by frequency and damping
    n_poles = len(all_poles)
    clustered = np.zeros(n_poles, dtype=bool)
    cluster_poles_out = []
    cluster_freqs_out = []
    cluster_damps_out = []

    for i in range(n_poles):
        if clustered[i]:
            continue

        # Find nearby poles
        freq_match = np.abs(all_freqs - all_freqs[i]) < freq_tol_hz
        damp_match = np.abs(all_damps - all_damps[i]) < damping_tol
        cluster_mask = freq_match & damp_match

        n_cluster = np.sum(cluster_mask)

        # Keep clusters with ≥2 members (stable across combinations)
        if n_cluster >= 2:
            # Take centroid of cluster
            cluster_poles_out.append(np.mean(all_poles[cluster_mask]))
            cluster_freqs_out.append(np.mean(all_freqs[cluster_mask]))
            cluster_damps_out.append(np.mean(all_damps[cluster_mask]))
            clustered[cluster_mask] = True

    return (np.array(cluster_poles_out),
            np.array(cluster_freqs_out),
            np.array(cluster_damps_out))


def filter_poles(poles: np.ndarray, frequencies: np.ndarray, damping_ratios: np.ndarray,
                 max_damping: float = 0.2, min_freq: float = 0.0, max_freq: float = np.inf) -> np.ndarray:
    """
    Filter poles based on physical criteria.

    Args:
        poles: Complex poles
        frequencies: Natural frequencies in Hz
        damping_ratios: Damping ratios
        max_damping: Maximum acceptable damping ratio
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz

    Returns:
        mask: Boolean mask of valid poles
    """
    # Keep poles with reasonable damping
    valid_damping = np.abs(damping_ratios) < max_damping

    # Keep poles in frequency range
    valid_freq = (frequencies >= min_freq) & (frequencies <= max_freq)

    # Keep poles with positive imaginary part (avoid conjugate duplicates)
    valid_imag = poles.imag > 0

    return valid_damping & valid_freq & valid_imag


def esprit_modal_identification(signals: np.ndarray, fs: float,
                                 model_order: int,
                                 window_length: Optional[int] = None,
                                 use_gpu: bool = False,
                                 max_damping: float = 0.2,
                                 freq_range: Tuple[float, float] = (0, np.inf),
                                 ref_sensor: int = 0,
                                 use_stabilization: bool = False) -> ModalParameters:
    """
    Complete ESPRIT modal identification from multi-channel decay signals.

    Args:
        signals: Multi-channel decay signals, shape (T, n_channels)
        fs: Sampling frequency in Hz
        model_order: Model order M (number of poles to extract)
        window_length: Hankel matrix row dimension L (default: T//2)
        use_gpu: Use GPU acceleration if available
        max_damping: Maximum acceptable damping ratio for filtering
        freq_range: (min_freq, max_freq) in Hz for pole filtering
        ref_sensor: Reference sensor for mode shape normalization
        use_stabilization: Enable automatic (M, L) grid stabilization (default: False)

    Returns:
        ModalParameters object containing poles, frequencies, damping, and shapes
    """
    T, n_channels = signals.shape
    dt = 1.0 / fs

    # Default window length: ~50% of signal length
    if window_length is None:
        window_length = T // 2

    # Stabilization grid: try multiple (M, L) combinations
    if use_stabilization:
        # Define grid (small for efficiency)
        L_candidates = [L for L in [1024, 1536, 2000, window_length] if L < T]
        M_candidates = [20, 30, 40, model_order]

        poles_list = []
        freqs_list = []
        damps_list = []

        for L in L_candidates:
            for M in M_candidates:
                try:
                    H = build_hankel_matrix(signals[:, 0], L)
                    poles_trial, _ = esprit_poles(H, M, dt, use_gpu=use_gpu)
                    freqs_trial, damps_trial = poles_to_modal_params(poles_trial, fs)

                    # Filter invalid poles
                    mask = filter_poles(poles_trial, freqs_trial, damps_trial,
                                       max_damping=max_damping,
                                       min_freq=freq_range[0],
                                       max_freq=freq_range[1])

                    if np.any(mask):
                        poles_list.append(poles_trial[mask])
                        freqs_list.append(freqs_trial[mask])
                        damps_list.append(damps_trial[mask])
                except:
                    pass  # Skip failed combinations

        # Cluster poles that appear in ≥2 combinations
        poles, frequencies, damping_ratios = cluster_poles(
            poles_list, freqs_list, damps_list,
            freq_tol_hz=5.0, damping_tol=0.02
        )

        # Use last singular values (from largest L, M)
        H = build_hankel_matrix(signals[:, 0], max(L_candidates))
        _, singular_values = esprit_poles(H, max(M_candidates), dt, use_gpu=use_gpu)

    else:
        # Single (M, L) combination (original behavior)
        H = build_hankel_matrix(signals[:, 0], window_length)

        # Extract poles using ESPRIT
        poles_all, singular_values = esprit_poles(H, model_order, dt, use_gpu=use_gpu)

        # Convert to frequencies and damping
        frequencies, damping_ratios = poles_to_modal_params(poles_all, fs)

        # Filter poles
        mask = filter_poles(poles_all, frequencies, damping_ratios,
                           max_damping=max_damping,
                           min_freq=freq_range[0],
                           max_freq=freq_range[1])

        poles = poles_all[mask]
        frequencies = frequencies[mask]
        damping_ratios = damping_ratios[mask]

    # Estimate mode shapes using all channels
    mode_shapes_raw = estimate_mode_shapes(signals, poles, dt)

    # Adaptive reference sensor selection: use channel with highest variance (best SNR)
    # This avoids normalizing against a weak/noisy channel
    if ref_sensor == 0 and n_channels > 1:  # Only auto-select if using default
        ref_sensor_auto = int(np.argmax(np.std(signals, axis=0)))
    else:
        ref_sensor_auto = ref_sensor

    # Normalize mode shapes
    mode_shapes = normalize_mode_shapes(mode_shapes_raw, ref_sensor=ref_sensor_auto)

    return ModalParameters(
        poles=poles,
        frequencies=frequencies,
        damping_ratios=damping_ratios,
        mode_shapes=mode_shapes,
        model_order=model_order,
        singular_values=singular_values
    )


def reconstruct_signal(mode_shapes: np.ndarray, poles: np.ndarray,
                       n_samples: int, dt: float) -> np.ndarray:
    """
    Reconstruct multi-channel signal from modal parameters.

    Args:
        mode_shapes: Complex mode shapes, shape (M, n_channels)
        poles: Complex poles, shape (M,)
        n_samples: Number of time samples to reconstruct
        dt: Time step in seconds

    Returns:
        reconstructed: Reconstructed signals, shape (n_samples, n_channels)
    """
    M, n_channels = mode_shapes.shape
    n = np.arange(n_samples)

    # Build modal exponentials
    Z = np.exp(np.outer(n, poles) * dt)  # shape (n_samples, M)

    # Reconstruct each channel
    reconstructed = Z @ mode_shapes  # shape (n_samples, n_channels)

    return reconstructed.real  # Take real part for physical signal
