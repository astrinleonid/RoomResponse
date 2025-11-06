"""
Signal Processor Module

Extracted signal processing logic from RoomResponseRecorder for better modularity.
This module provides pure signal processing operations for impulse response measurements.

Responsibilities:
- Cycle extraction and reshaping
- Signal alignment (standard and calibration modes)
- Normalization and averaging
- Spectral analysis
- Onset detection

This code was extracted from RoomResponseRecorder.py to separate recording orchestration
from signal processing logic, improving testability and reusability.

Created: 2025-11-03
Extracted from: RoomResponseRecorder.py (commit bb0580e)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class SignalProcessingConfig:
    """
    Configuration for signal processing operations.

    This holds all parameters needed by SignalProcessor, extracted from
    RoomResponseRecorder to enable standalone signal processing.
    """
    num_pulses: int          # Number of cycles/pulses in recording
    cycle_samples: int       # Samples per cycle
    sample_rate: int         # Audio sample rate (Hz)
    multichannel_config: Dict[str, Any] = None  # Multi-channel configuration

    def __post_init__(self):
        """Initialize multichannel_config if not provided"""
        if self.multichannel_config is None:
            self.multichannel_config = {}


class SignalProcessor:
    """
    Signal processing operations for impulse response measurements.

    Handles cycle extraction, alignment, normalization, averaging, and spectral analysis.
    Separated from recording logic for better modularity and testability.

    All methods are pure signal processing - they take audio data as input and return
    processed results. No hardware or file I/O dependencies.
    """

    def __init__(self, config: SignalProcessingConfig):
        """
        Initialize signal processor with configuration.

        Args:
            config: Signal processing configuration
        """
        self.config = config

    # ========================================================================
    # UNIVERSAL METHODS (used by all modes)
    # ========================================================================

    def extract_cycles(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract cycles from raw audio using simple reshape.

        Pads or trims audio to expected length, then reshapes into cycles.

        Args:
            audio: Raw audio signal

        Returns:
            Cycles array [num_cycles, cycle_samples]
        """
        expected_samples = self.config.cycle_samples * self.config.num_pulses

        # Pad or trim to expected length
        if len(audio) < expected_samples:
            padded = np.zeros(expected_samples, dtype=audio.dtype)
            padded[:len(audio)] = audio
            audio = padded
        else:
            audio = audio[:expected_samples]

        # Reshape into cycles
        return audio.reshape(self.config.num_pulses, self.config.cycle_samples)

    def average_cycles(self, cycles: np.ndarray, start_cycle: int = None) -> np.ndarray:
        """
        Average cycles starting from start_cycle.

        Skips initial cycles to allow system settling.

        Args:
            cycles: Cycles array [num_cycles, cycle_samples]
            start_cycle: Index to start averaging from (default: num_pulses // 4)

        Returns:
            Averaged signal [cycle_samples]
        """
        if start_cycle is None:
            start_cycle = max(1, self.config.num_pulses // 4)

        return np.mean(cycles[start_cycle:], axis=0)

    def compute_spectral_analysis(self,
                                    responses: Dict[int, np.ndarray],
                                    window_start: float = 0.0,
                                    window_end: float = 1.0) -> Dict[str, Any]:
        """
        Compute FFT spectral analysis of responses.

        Args:
            responses: Response signals per channel (Dict[ch_idx -> np.ndarray])
            window_start: Start of analysis window (fraction 0.0-1.0)
            window_end: End of analysis window (fraction 0.0-1.0)

        Returns:
            Dict with:
                'frequencies': np.ndarray - Frequency bins (Hz)
                'magnitudes': Dict[ch_idx -> np.ndarray] - FFT magnitude per channel
                'magnitude_db': Dict[ch_idx -> np.ndarray] - Magnitude in dB per channel
                'window': [start_frac, end_frac] - Window used
                'n_fft': int - FFT size
        """
        # Get first response to determine length
        first_response = next(iter(responses.values()))
        n_samples = len(first_response)

        # Apply window
        start_idx = int(window_start * n_samples)
        end_idx = int(window_end * n_samples)

        # Compute FFT for each channel
        frequencies = np.fft.rfftfreq(n_samples, 1 / self.config.sample_rate)
        magnitudes = {}
        magnitude_db = {}

        for ch_idx, response in responses.items():
            windowed = response[start_idx:end_idx]
            fft = np.fft.rfft(windowed, n=n_samples)
            mag = np.abs(fft)
            magnitudes[ch_idx] = mag
            magnitude_db[ch_idx] = 20 * np.log10(mag + 1e-10)

        return {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'magnitude_db': magnitude_db,
            'window': [window_start, window_end],
            'n_fft': n_samples
        }

    def find_sound_onset(self, audio: np.ndarray, window_size: int = 10,
                          threshold_factor: float = 2) -> int:
        """Find sound onset using moving average and derivative"""
        if len(audio) < window_size * 2:
            return 0

        # Calculate moving RMS
        def moving_rms(signal, window):
            padded = np.pad(signal, window // 2, mode='constant')
            return np.sqrt(np.convolve(padded ** 2, np.ones(window) / window, mode='valid'))

        rms = moving_rms(audio, window_size)
        rms_diff = np.diff(rms)

        # Find significant increase
        background_level = np.std(rms[:window_size]) if len(rms) > window_size else np.std(rms)
        threshold = threshold_factor * background_level

        onset_candidates = np.where(rms_diff > threshold)[0]
        return onset_candidates[0] if len(onset_candidates) > 0 else 0

    # ========================================================================
    # STANDARD MODE METHODS
    # ========================================================================

    def find_onset_in_room_response(self, room_response: np.ndarray) -> int:
        """
        Find onset position in a room response (extracted helper method)
        """
        max_index = np.argmax(np.abs(room_response))

        if max_index > 50:
            search_start = max(0, max_index - 100)
            search_window = room_response[search_start:max_index + 50]
            onset_in_window = self.find_sound_onset(search_window)
            onset = search_start + onset_in_window
        else:
            onset = 0

        return onset

    def extract_impulse_response(self, room_response: np.ndarray) -> np.ndarray:
        """Extract impulse response by finding onset and rotating signal"""
        try:
            max_index = np.argmax(np.abs(room_response))

            if max_index > 50:  # Search for onset if peak not at beginning
                search_start = max(0, max_index - 100)
                search_window = room_response[search_start:max_index + 50]

                onset_in_window = self.find_sound_onset(search_window)
                onset = search_start + onset_in_window

                print(f"Found onset at sample {onset} (peak at {max_index})")

                # Rotate signal to put onset at beginning
                impulse_response = np.concatenate([room_response[onset:], room_response[:onset]])
            else:
                print("Peak near beginning, using room response as impulse response")
                impulse_response = room_response.copy()

            return impulse_response

        except Exception as e:
            print(f"Error extracting impulse response: {e}")
            return room_response.copy()

    # ========================================================================
    # CALIBRATION MODE METHODS (Advanced Per-Cycle Alignment)
    # ========================================================================
    #
    # NOTE ON ALIGNMENT SYSTEMS:
    #
    # This is the ADVANCED alignment system for calibration mode, designed for
    # physical impacts with variable timing (e.g., piano hammer strikes).
    #
    # KEY DIFFERENCES FROM STANDARD MODE:
    # 1. PER-CYCLE alignment - Each cycle is aligned individually
    # 2. QUALITY FILTERING - Only validated cycles are used
    # 3. CROSS-CORRELATION - Outliers are removed based on similarity
    # 4. ROBUST to timing variance - Handles ±several samples of jitter
    #
    # Standard mode uses SIMPLE alignment (see find_onset_in_room_response):
    # - Single onset detection in averaged response
    # - Fast and efficient for synthetic pulses
    # - Assumes minimal timing jitter
    #
    # Both are INTENTIONAL - choose based on signal source characteristics.
    # ========================================================================

    def align_cycles_by_onset(self, initial_cycles: np.ndarray, validation_results: list,
                             correlation_threshold: float = 0.7) -> dict:
        """
        STEP 5: Align cycles by detecting onset (negative peak) in each cycle.

        Process:
        1. Filter: Keep only VALID cycles (from validation)
        2. Find onset: Locate negative peak in each valid cycle
        3. Align: Shift all cycles so negative peaks align at same position
        4. Cross-correlation check: Verify aligned cycles correlate well
        5. Filter again: Remove cycles with poor correlation after alignment

        Args:
            initial_cycles: 2D array from simple reshape (num_cycles, cycle_samples)
            validation_results: List of validation dicts from CalibrationValidatorV2
            correlation_threshold: Minimum correlation after alignment (default 0.7)

        Returns:
            Dictionary containing:
                - 'aligned_cycles': 2D array of aligned, filtered cycles
                - 'valid_cycle_indices': Original indices of cycles kept
                - 'onset_positions': Onset position found in each original cycle
                - 'aligned_onset_position': Common onset position in aligned cycles
                - 'correlations': Cross-correlation values after alignment
                - 'reference_cycle_idx': Index of reference cycle (in valid set)
        """
        if len(initial_cycles) == 0:
            return {
                'aligned_cycles': np.array([]),
                'valid_cycle_indices': [],
                'onset_positions': [],
                'aligned_onset_position': 0,
                'correlations': [],
                'reference_cycle_idx': 0
            }

        # STEP 1: Filter - Keep only VALID cycles
        valid_indices = [i for i, v in enumerate(validation_results) if v.get('calibration_valid', False)]

        if len(valid_indices) == 0:
            # No valid cycles
            return {
                'aligned_cycles': np.array([]),
                'valid_cycle_indices': [],
                'onset_positions': [],
                'aligned_onset_position': 0,
                'correlations': [],
                'reference_cycle_idx': 0
            }

        valid_cycles = initial_cycles[valid_indices]

        # STEP 2: Find onset (negative peak) in each valid cycle
        onset_positions = []
        for cycle in valid_cycles:
            # Find index of negative peak (minimum value)
            onset_idx = int(np.argmin(cycle))
            onset_positions.append(onset_idx)

        # STEP 3: Determine common onset position
        # Default: 0 (onset at beginning of waveform)
        # User can override to preserve pre-onset data (e.g., 100 samples for ~2ms buffer)
        target_onset_position = self.config.multichannel_config.get('alignment_target_onset_position', 0)
        aligned_onset_position = target_onset_position

        # STEP 4: Align all cycles by shifting to common onset position
        aligned_cycles_list = []
        for i, cycle in enumerate(valid_cycles):
            shift_needed = aligned_onset_position - onset_positions[i]

            # Apply circular shift
            aligned_cycle = np.roll(cycle, shift_needed)
            aligned_cycles_list.append(aligned_cycle)

        aligned_cycles = np.array(aligned_cycles_list)

        # STEP 5: Select reference (highest energy among aligned)
        energies = np.sqrt(np.mean(aligned_cycles ** 2, axis=1))
        reference_idx = int(np.argmax(energies))
        reference_cycle = aligned_cycles[reference_idx]

        # STEP 6: Calculate cross-correlation with reference
        correlations = []
        for i, cycle in enumerate(aligned_cycles):
            if i == reference_idx:
                correlations.append(1.0)
            else:
                # Compute normalized cross-correlation at zero lag
                ref_energy = np.sum(reference_cycle ** 2)
                cyc_energy = np.sum(cycle ** 2)
                cross_product = np.sum(reference_cycle * cycle)

                if ref_energy > 0 and cyc_energy > 0:
                    corr_value = float(cross_product / np.sqrt(ref_energy * cyc_energy))
                else:
                    corr_value = 0.0

                correlations.append(corr_value)

        # STEP 7: Filter by correlation threshold
        final_indices = [i for i, corr in enumerate(correlations) if corr >= correlation_threshold]

        if len(final_indices) == 0:
            # All cycles filtered out
            return {
                'aligned_cycles': np.array([]),
                'valid_cycle_indices': [],
                'onset_positions': onset_positions,
                'aligned_onset_position': aligned_onset_position,
                'correlations': correlations,
                'reference_cycle_idx': reference_idx
            }

        # Return only cycles that passed correlation threshold
        final_aligned_cycles = aligned_cycles[final_indices]
        final_valid_indices = [valid_indices[i] for i in final_indices]
        final_correlations = [correlations[i] for i in final_indices]
        final_onset_positions = [onset_positions[i] for i in final_indices]

        # Adjust reference index to final set
        if reference_idx in final_indices:
            final_reference_idx = final_indices.index(reference_idx)
        else:
            final_reference_idx = 0

        return {
            'aligned_cycles': final_aligned_cycles,
            'valid_cycle_indices': final_valid_indices,
            'onset_positions': final_onset_positions,  # Only for cycles that passed correlation
            'aligned_onset_position': aligned_onset_position,
            'correlations': final_correlations,
            'reference_cycle_idx': final_reference_idx,
            'correlation_threshold': correlation_threshold
        }

    def apply_alignment_to_channel(self, channel_raw: np.ndarray,
                                   alignment_metadata: dict) -> np.ndarray:
        """
        Apply alignment shifts (calculated from calibration channel) to any channel.

        This ensures all channels are aligned uniformly based on calibration channel timing.

        Args:
            channel_raw: Raw audio from any channel (1D array)
            alignment_metadata: Alignment metadata from align_cycles_by_onset()

        Returns:
            2D array of aligned cycles (num_valid_cycles, cycle_samples)
            Only returns cycles that passed validation and correlation filters.
        """
        # Extract alignment info
        valid_cycle_indices = alignment_metadata.get('valid_cycle_indices', [])
        onset_positions = alignment_metadata.get('onset_positions', [])
        aligned_onset_position = alignment_metadata.get('aligned_onset_position', 0)

        if len(valid_cycle_indices) == 0:
            return np.array([])

        # Pad or trim channel to expected length
        expected_samples = self.config.cycle_samples * self.config.num_pulses
        if len(channel_raw) < expected_samples:
            padded = np.zeros(expected_samples)
            padded[:len(channel_raw)] = channel_raw
            channel_raw = padded
        else:
            channel_raw = channel_raw[:expected_samples]

        # Extract initial cycles using simple reshape
        initial_cycles = channel_raw.reshape(self.config.num_pulses, self.config.cycle_samples)

        # Apply the SAME shifts to this channel's cycles
        aligned_cycles_list = []
        for i, original_idx in enumerate(valid_cycle_indices):
            if original_idx < len(initial_cycles):
                cycle = initial_cycles[original_idx]

                # Calculate shift (same logic as calibration channel)
                if i < len(onset_positions):
                    original_onset = onset_positions[i]
                    shift_needed = aligned_onset_position - original_onset

                    # Apply circular shift
                    aligned_cycle = np.roll(cycle, shift_needed)
                    aligned_cycles_list.append(aligned_cycle)

        return np.array(aligned_cycles_list)

    def normalize_by_calibration(self,
                                   aligned_multichannel_cycles: Dict[int, np.ndarray],
                                   validation_results: List[Dict],
                                   calibration_channel: int,
                                   valid_cycle_indices: List[int]) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """
        Normalize response channels by calibration signal magnitude.

        Each response channel's amplitude is divided by the corresponding calibration
        impulse magnitude (negative peak) to produce calibration-normalized responses.
        This enables quantitative comparison across measurements and removes impact
        strength variability.

        Args:
            aligned_multichannel_cycles: Dict[channel_idx -> array[n_cycles, samples]]
                                         Aligned cycles from all channels
            validation_results: List of validation dicts with calibration_metrics
                               containing 'negative_peak' values
            calibration_channel: Channel index of calibration sensor
            valid_cycle_indices: List of original cycle indices for aligned cycles
                                (maps aligned_idx -> original_idx)

        Returns:
            Tuple of:
            - Dict[channel_idx -> normalized array[n_cycles, samples]]: Normalized cycles
            - List[float]: Normalization factors (negative peak magnitudes) for each cycle

        Notes:
            - Calibration channel is excluded from normalization (kept as raw aligned)
            - Response channels are divided by |negative_peak| for each cycle
            - Returns empty dict if no valid cycles or missing negative peaks
            - Protects against division by zero (skips cycles with peak < 1e-6)
        """
        # Extract negative peak values for each cycle
        normalization_factors = []
        for v_result in validation_results:
            metrics = v_result.get('calibration_metrics') or {}
            if not isinstance(metrics, dict):
                metrics = {}
            neg_peak = abs(metrics.get('negative_peak', 0.0))
            normalization_factors.append(neg_peak)

        # Validate we have normalization factors
        if not normalization_factors:
            print("Warning: No normalization factors available")
            return {}, []

        # Get valid cycle indices from first channel's aligned data
        if not aligned_multichannel_cycles:
            return {}, normalization_factors

        first_channel_data = next(iter(aligned_multichannel_cycles.values()))
        num_aligned_cycles = len(first_channel_data)

        # Ensure we have enough normalization factors
        if len(normalization_factors) < num_aligned_cycles:
            print(f"Warning: Mismatch between aligned cycles ({num_aligned_cycles}) "
                  f"and normalization factors ({len(normalization_factors)})")
            # Use only available factors
            normalization_factors = normalization_factors[:num_aligned_cycles]

        # Normalize each response channel
        normalized_multichannel_cycles = {}

        for ch_idx, channel_cycles in aligned_multichannel_cycles.items():
            if ch_idx == calibration_channel:
                # Keep calibration channel unnormalized
                normalized_multichannel_cycles[ch_idx] = channel_cycles.copy()
                continue

            # Normalize response channel
            normalized_cycles = []
            for cycle_idx, cycle_data in enumerate(channel_cycles):
                # Map aligned cycle index to original cycle index
                if cycle_idx < len(valid_cycle_indices):
                    original_idx = valid_cycle_indices[cycle_idx]

                    # Get normalization factor for the original cycle
                    if original_idx < len(normalization_factors):
                        norm_factor = normalization_factors[original_idx]

                        # Protect against division by zero
                        if norm_factor > 1e-6:
                            normalized_cycle = cycle_data / norm_factor
                            normalized_cycles.append(normalized_cycle)
                        else:
                            print(f"Warning: Skipping cycle {original_idx} (aligned position {cycle_idx}) "
                                  f"in channel {ch_idx} (normalization factor too small: {norm_factor})")
                            # Keep unnormalized if factor is too small
                            normalized_cycles.append(cycle_data)
                    else:
                        print(f"Warning: Original cycle index {original_idx} out of range for "
                              f"normalization factors (len={len(normalization_factors)})")
                        normalized_cycles.append(cycle_data)
                else:
                    print(f"Warning: Aligned cycle {cycle_idx} out of range for "
                          f"valid_cycle_indices (len={len(valid_cycle_indices)})")
                    normalized_cycles.append(cycle_data)

            normalized_multichannel_cycles[ch_idx] = np.array(normalized_cycles)

        return normalized_multichannel_cycles, normalization_factors

    # ========================================================================
    # TRUNCATION METHODS
    # ========================================================================

    def truncate_with_fadeout(
        self,
        signal: np.ndarray,
        working_length_ms: float,
        fade_length_ms: float
    ) -> np.ndarray:
        """
        Truncate signal to working length with fade-out envelope.

        Reduces the length of impulse responses while avoiding abrupt cutoffs
        by applying a smooth fade-out window at the end.

        Args:
            signal: Input signal (1D array)
            working_length_ms: Duration to keep in milliseconds
            fade_length_ms: Fade-out window duration in milliseconds

        Returns:
            Truncated signal with fade-out applied

        Raises:
            ValueError: If working_length_ms <= fade_length_ms or if parameters are invalid
        """
        # Convert durations to samples
        working_samples = int(working_length_ms * self.config.sample_rate / 1000.0)
        fade_samples = int(fade_length_ms * self.config.sample_rate / 1000.0)

        # Validate parameters
        if fade_samples <= 0:
            raise ValueError(f"Fade length must be positive, got {fade_length_ms} ms")

        if working_samples <= fade_samples:
            raise ValueError(
                f"Working length ({working_length_ms} ms = {working_samples} samples) "
                f"must be greater than fade length ({fade_length_ms} ms = {fade_samples} samples)"
            )

        # If signal already shorter than target, no truncation needed
        if working_samples >= len(signal):
            return signal

        # Truncate to working length
        truncated = signal[:working_samples].copy()

        # Apply fade-out envelope using Hann window (second half for smooth fade)
        # Hann window provides smoother fade than linear
        fade_start = working_samples - fade_samples
        fade_window = np.hanning(fade_samples * 2)[fade_samples:]  # Second half of Hann
        truncated[fade_start:] *= fade_window

        return truncated
