"""
Streaming version of esprit.py for real-time processing.

This module allows processing measurements one-by-one as they arrive,
following the exact implementation of esprit.py to guarantee identical results.
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json
from scipy.cluster.vq import kmeans, vq

# Import all functions from esprit.py (relative import from same directory)
from .esprit import (
    build_hankel,
    TLS_ESPRIT_FromUs,
    LambdasToFZQ,
    BandPreset,
    band_presets,
    read_index,
    process_cube
)


class StreamingESPRITProcessor:
    """
    Processes measurements one-by-one following esprit.py exactly.

    Usage:
        # Initialize
        processor = StreamingESPRITProcessor(
            M_out=6, N_use=28800, fs=48000,
            band_index=0, L_fraction=0.5, K=30
        )

        # Process measurements as they arrive
        for r_index, measurement in enumerate(measurements):
            result = processor.process_measurement(r_index, measurement)
            print(f"Processed point {r_index}: {len(result['frequencies'])} modes")

        # Get final aggregated results
        final_results = processor.get_final_results()
    """

    def __init__(self,
                 M_out: int,
                 N_use: int,
                 fs: float,
                 band_index: int = 0,
                 L_fraction: float = 0.5,
                 K: int = 30,
                 skip_m: Optional[int] = 2):
        """
        Initialize the streaming processor.

        Args:
            M_out: Number of measurement channels (e.g., 6)
            N_use: Number of samples in raw measurement
            fs: Sampling frequency (Hz)
            band_index: Which band preset to use (0-3)
            L_fraction: Hankel window length as fraction of signal
            K: Model order (number of poles to extract)
            skip_m: Channel index to skip (e.g., hammer calibration channel), or None to use all channels
        """
        self.M_out = M_out
        self.N_use = N_use
        self.fs = fs
        self.band_index = band_index
        self.L_fraction = L_fraction
        self.K = K
        self.skip_m = skip_m

        # Band processing parameters
        self.current_preset = band_presets[band_index]
        self.fs_band = fs / self.current_preset.decimate_factor
        self.dt = 1.0 / self.fs_band

        # Channel mapping (exclude skip_m if specified, otherwise use all channels)
        if skip_m is not None:
            self.m_map = [m for m in range(M_out) if m != skip_m]
        else:
            self.m_map = list(range(M_out))
        self.M_eff = len(self.m_map)

        # Storage for incremental results
        self.all_f = []
        self.all_z = []
        self.all_r = []
        self.all_lam_re = []
        self.all_lam_im = []

        # Storage for processed measurements (for mode shape fitting later)
        self.processed_measurements = []  # List of (r_index, processed_cube_r)
        self.r_indices = []

        print(f"StreamingESPRITProcessor initialized:")
        print(f"  Band: {self.current_preset.low_freq}-{self.current_preset.high_freq} Hz")
        print(f"  fs_band: {self.fs_band} Hz")
        print(f"  Effective channels: {self.M_eff}")
        print(f"  Model order K: {self.K}")

    def preprocess_single_measurement(self, y_raw: np.ndarray) -> np.ndarray:
        """
        Preprocess a single measurement following esprit.py exactly.

        Args:
            y_raw: Raw measurement array of shape (M_out, N_use)

        Returns:
            Processed measurement of shape (M_eff, N_band)
        """
        processed = np.zeros((self.M_eff, self.current_preset.N_band))

        for me in range(self.M_eff):
            m = self.m_map[me]
            y_orig = y_raw[m, :self.N_use]

            # Bandpass filter (exact copy from esprit.py lines 684-685)
            sos = scipy.signal.butter(
                self.current_preset.filter_order,
                [self.current_preset.low_freq / (self.fs / 2),
                 self.current_preset.high_freq / (self.fs / 2)],
                btype='bandpass',
                analog=False,
                output='sos'
            )
            y_band = scipy.signal.sosfilt(sos, y_orig)

            # Exponential window (exact copy from esprit.py lines 687-688)
            # Use actual signal length instead of N_use to handle varying sample sizes
            t = np.arange(len(y_band)) / self.fs
            y_band *= np.exp(-self.current_preset.exp_alpha * t)

            # Decimate (exact copy from esprit.py lines 690-691)
            if self.current_preset.decimate_factor > 1:
                y_band = scipy.signal.decimate(y_band, self.current_preset.decimate_factor, ftype='iir')

            # Truncate (exact copy from esprit.py lines 693-695)
            N_proc = len(y_band)
            N_band = min(N_proc, self.current_preset.N_band)
            processed[me, :N_band] = y_band[:N_band]

        return processed

    def process_measurement(self, r_index: int, y_raw: np.ndarray) -> dict:
        """
        Process a single measurement as it arrives.

        Args:
            r_index: Index of this excitation point
            y_raw: Raw measurement array of shape (M_out, N_use)

        Returns:
            Dictionary with extracted modes for this measurement:
                {
                    'r_index': int,
                    'frequencies': np.ndarray,
                    'damping_ratios': np.ndarray,
                    'Q_factors': np.ndarray,
                    'lambdas_re': np.ndarray,
                    'lambdas_im': np.ndarray
                }
        """
        print(f"\n[STREAM] Processing measurement r={r_index}")

        # Preprocess
        processed = self.preprocess_single_measurement(y_raw)
        N_band = processed.shape[1]

        # Store for later mode shape fitting
        self.processed_measurements.append((r_index, processed))
        self.r_indices.append(r_index)

        # Calculate Hankel matrix dimensions
        L = int(N_band * self.L_fraction)
        C = N_band - L + 1

        try:
            # Build stacked Hankel matrix (exact copy from esprit.py line 764)
            big_H = np.vstack([
                build_hankel(processed[me, :N_band], N_band, L)
                for me in range(self.M_eff)
            ])

            # SVD (exact copy from esprit.py lines 765-766)
            U, S, Vh = np.linalg.svd(big_H, full_matrices=False)
            V = Vh.T
            Us = U[:, :self.K]

            # TLS-ESPRIT from U subspace (exact copy from esprit.py line 768)
            lam_re, lam_im = TLS_ESPRIT_FromUs(Us, self.M_eff * L, L, self.M_eff, self.K)

            # Convert to frequencies and damping (exact copy from esprit.py line 769)
            F, Q, Z = LambdasToFZQ(lam_re, lam_im, self.K, self.dt)

            # Store results (exact copy from esprit.py lines 770-774)
            self.all_f.extend(F)
            self.all_z.extend(Z)
            self.all_r.extend([r_index] * len(F))
            self.all_lam_re.append(lam_re)
            self.all_lam_im.append(lam_im)

            print(f"[STREAM] Found {len(F)} modes for r={r_index}")

            return {
                'r_index': r_index,
                'frequencies': F,
                'damping_ratios': Z,
                'Q_factors': Q,
                'lambdas_re': lam_re,
                'lambdas_im': lam_im
            }

        except Exception as e:
            print(f"[STREAM] Error processing r={r_index}: {e}")
            return {
                'r_index': r_index,
                'frequencies': np.array([]),
                'damping_ratios': np.array([]),
                'Q_factors': np.array([]),
                'lambdas_re': np.array([]),
                'lambdas_im': np.array([]),
                'error': str(e)
            }

    def get_current_stabilization_plot(self):
        """
        Generate stabilization diagram with current results.
        Can be called at any time to see progress.
        """
        if len(self.all_f) == 0:
            print("No modes found yet")
            return

        plt.figure(figsize=(12, 6))
        plt.scatter(self.all_r, self.all_f, s=5, alpha=0.6)
        plt.xlabel('Excitation Point (r)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Stabilization Diagram (Current: {len(self.r_indices)} measurements)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_final_results(self, num_clusters: Optional[int] = None,
                         names: Optional[List[str]] = None,
                         output_file: Optional[str] = None,
                         min_occurrence_pct: float = 30.0,
                         freq_tolerance_pct: float = 2.0) -> dict:
        """
        Compute final aggregated results after all measurements are processed.

        Args:
            num_clusters: Number of mode clusters (default: auto based on data)
            names: Optional names for excitation points
            output_file: Optional path to save JSON results
            min_occurrence_pct: Minimum % of scenarios a mode must appear in (default: 30%)
            freq_tolerance_pct: Frequency tolerance for grouping modes (default: 2%)

        Returns:
            Dictionary with:
                'common_f': List of common mode frequencies
                'common_z': List of common mode damping ratios
                'signed_shapes': List of mode shapes along r
                'participation': List of participation factors
                'amplitudes_in_m': List of amplitudes in receivers
                'r_indices': List of r indices
                'names': List of names (if provided)
                'mode_occurrences': List of occurrence counts per mode
        """
        R = len(self.r_indices)
        print(f"\n[FINAL] Computing final results from {R} measurements")
        print(f"[FINAL] Criteria: min_occurrence={min_occurrence_pct}%, freq_tolerance={freq_tolerance_pct}%")

        if len(self.all_f) == 0:
            print("No modes found")
            return {}

        # Convert to arrays for easier manipulation
        all_f = np.array(self.all_f)
        all_z = np.array(self.all_z)
        all_r = np.array(self.all_r)

        # Use adaptive clustering based on frequency tolerance
        # Group modes that are within freq_tolerance_pct of each other
        sorted_indices = np.argsort(all_f)
        sorted_f = all_f[sorted_indices]
        sorted_z = all_z[sorted_indices]
        sorted_r = all_r[sorted_indices]

        # Find mode groups using frequency tolerance
        mode_groups = []
        current_group = [0]

        for i in range(1, len(sorted_f)):
            # Check if this frequency is within tolerance of the group's mean
            group_mean = np.mean(sorted_f[current_group])
            tolerance = group_mean * freq_tolerance_pct / 100.0

            if abs(sorted_f[i] - group_mean) <= tolerance:
                current_group.append(i)
            else:
                mode_groups.append(current_group)
                current_group = [i]

        if current_group:
            mode_groups.append(current_group)

        print(f"[FINAL] Found {len(mode_groups)} potential mode groups")

        # Filter groups by minimum occurrence
        min_occurrences = int(R * min_occurrence_pct / 100.0)
        min_occurrences = max(1, min_occurrences)  # At least 1

        common_f = []
        common_z = []
        mode_occurrences = []

        for group in mode_groups:
            # Count unique scenarios where this mode appears
            scenarios_in_group = set(sorted_r[group])
            occurrence_count = len(scenarios_in_group)

            if occurrence_count >= min_occurrences:
                mean_f = np.mean(sorted_f[group])
                mean_z = np.mean(sorted_z[group])

                if not np.isnan(mean_f) and not np.isnan(mean_z) and mean_z > 0:
                    common_f.append(mean_f)
                    common_z.append(mean_z)
                    mode_occurrences.append(occurrence_count)

        # Sort by frequency
        if len(common_f) > 0:
            sort_idx = np.argsort(common_f)
            common_f = [common_f[i] for i in sort_idx]
            common_z = [common_z[i] for i in sort_idx]
            mode_occurrences = [mode_occurrences[i] for i in sort_idx]

        print(f"[FINAL] Common modes (appearing in >={min_occurrences}/{R} scenarios): {len(common_f)}")
        for i in range(len(common_f)):
            pct = mode_occurrences[i] / R * 100
            print(f"  Mode {i}: f={common_f[i]:.2f} Hz, zeta={common_z[i]:.4f}, occurs in {mode_occurrences[i]}/{R} ({pct:.0f}%)")

        # Fit mode shapes (exact copy from esprit.py lines 814-857)
        R = len(self.r_indices)

        # Get N_band from first processed measurement
        N_band = self.processed_measurements[0][1].shape[1]
        t = np.arange(N_band) * self.dt

        mode_shapes_along_r = []
        amplitudes_in_m = []
        participation = []
        signed_shapes = []

        for k in range(len(common_f)):
            f_k = common_f[k]
            zeta_k = common_z[k]

            if np.isnan(zeta_k) or zeta_k < 0 or zeta_k >= 1:
                continue

            w_k = 2 * np.pi * f_k
            if zeta_k < 1:
                alpha_k = zeta_k * w_k / np.sqrt(1 - zeta_k**2)
            else:
                alpha_k = zeta_k * w_k

            exp_a = np.exp(-alpha_k * t[:N_band])
            cos_term = exp_a * np.cos(w_k * t[:N_band])
            sin_term = exp_a * np.sin(w_k * t[:N_band])

            A_r_complex = np.zeros(R, dtype=complex)
            A_m_per_r = np.zeros((R, self.M_eff), dtype=complex)

            for ri in range(R):
                r_index, processed = self.processed_measurements[ri]
                A_me = np.zeros(self.M_eff, dtype=complex)

                for me in range(self.M_eff):
                    y = processed[me, :N_band]
                    basis = np.column_stack([cos_term, sin_term])
                    try:
                        coeff = np.linalg.lstsq(basis, y, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        coeff = np.zeros(2)

                    a, b = coeff if len(coeff) == 2 else (0.0, 0.0)
                    A_me[me] = a + 1j * b

                A_r_complex[ri] = np.mean(A_me)
                A_m_per_r[ri, :] = A_me

            mode_shapes_along_r.append(A_r_complex)
            amplitudes_in_m.append(np.mean(A_m_per_r, axis=0))
            participation.append(np.abs(A_r_complex))

            # Compute signed real shape (exact copy from esprit.py lines 851-856)
            if np.any(A_r_complex != 0):
                ref_phase = np.angle(A_r_complex[np.argmax(np.abs(A_r_complex))])
                signed_shape = np.real(A_r_complex * np.exp(-1j * ref_phase))
                signed_shape /= np.max(np.abs(signed_shape)) if np.max(np.abs(signed_shape)) != 0 else 1
            else:
                signed_shape = np.zeros(R)
            signed_shapes.append(signed_shape)

        # Prepare results dictionary
        results = {
            'common_f': common_f,
            'common_z': common_z,
            'mode_occurrences': mode_occurrences,
            'num_scenarios': R,
            'signed_shapes': [s.tolist() for s in signed_shapes],
            'participation': [p.tolist() for p in participation],
            'amplitudes_in_m': [[{"real": c.real, "imag": c.imag} for c in a] for a in amplitudes_in_m],
            'r_indices': self.r_indices
        }

        if names is not None:
            results['names'] = names

        # Save to file if requested
        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"[FINAL] Results saved to {output_file}")

        return results

    def plot_mode_shapes(self, results: dict):
        """
        Interactive mode shape viewer (exact copy from esprit.py lines 874-905).

        Args:
            results: Dictionary returned by get_final_results()
        """
        common_f = results['common_f']
        signed_shapes = [np.array(s) for s in results['signed_shapes']]
        r_indices = results['r_indices']
        names = results.get('names', [str(r) for r in r_indices])

        if len(common_f) == 0:
            print("No modes to plot")
            return

        class ModeViewer:
            def __init__(self, common_f, signed_shapes, names, r_points):
                self.common_f = common_f
                self.signed_shapes = signed_shapes
                self.names = names
                self.r_points = r_points
                self.k = 0
                self.fig, self.ax = plt.subplots(figsize=(12, 6))
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
                self.ax.plot(self.r_points, self.signed_shapes[self.k], 'o-', label='Signed Amplitude')
                self.ax.set_xlabel('Excitation points')
                self.ax.set_ylabel('Signed Amplitude')
                self.ax.set_title(f'Mode {self.k}: f={self.common_f[self.k]:.2f} Hz - Use arrow keys to navigate')
                self.ax.set_xticks(self.r_points)
                self.ax.set_xticklabels(self.names, rotation=45, ha='right')
                self.ax.grid(True, alpha=0.3)
                self.fig.tight_layout()
                self.fig.canvas.draw()

        r_points = np.arange(len(r_indices))
        viewer = ModeViewer(common_f, signed_shapes, names, r_points)
        plt.show()
