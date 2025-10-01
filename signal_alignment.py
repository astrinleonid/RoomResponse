#!/usr/bin/env python3
"""
Signal Alignment Module - Synchronize impulse responses across series

This module provides tools for aligning multiple audio signals (typically impulse responses)
by finding optimal time shifts through cross-correlation analysis.

Key features:
- Normalize signals to unit peak amplitude
- Threshold-based noise removal
- Cross-correlation based alignment
- Circular shift for periodic signals
- Batch processing with progress tracking

Usage:
    from signal_alignment import SignalAligner

    aligner = SignalAligner(threshold=0.3)
    results = aligner.align_signals(file_paths, reference_index=0)
    aligner.save_aligned_signals(results, output_dir)
"""

import os
import wave
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    """Results from signal alignment process."""
    original_signal: np.ndarray
    aligned_signal: np.ndarray
    shift_samples: int
    correlation_peak: float
    file_path: str
    sample_rate: int


class SignalAligner:
    """
    Aligns multiple audio signals using cross-correlation.

    Process:
    1. Normalize each signal (peak = 1.0)
    2. Apply threshold to remove noise
    3. Select reference signal
    4. Find optimal shift via cross-correlation
    5. Apply circular shift to align signals
    """

    def __init__(self, threshold: float = 0.3, normalize: bool = True):
        """
        Initialize signal aligner.

        Args:
            threshold: Amplitude threshold for noise removal (0.0-1.0)
            normalize: Whether to normalize signals before processing
        """
        self.threshold = threshold
        self.normalize = normalize

    def align_signals(
        self,
        file_paths: List[str],
        reference_index: int = 0,
        max_shift: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[AlignmentResult]:
        """
        Align multiple audio signals to a reference signal.

        Args:
            file_paths: List of paths to audio files
            reference_index: Index of reference signal (default: 0 = first file)
            max_shift: Maximum shift in samples (None = signal_length // 2)
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of AlignmentResult objects
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        if reference_index < 0 or reference_index >= len(file_paths):
            raise ValueError(f"Invalid reference_index: {reference_index}")

        # Load all signals
        signals = []
        sample_rate = None

        if progress_callback:
            progress_callback(0, len(file_paths), "Loading audio files...")

        for i, file_path in enumerate(file_paths):
            try:
                audio_data, sr = self._load_wav_file(file_path)

                if audio_data is None or len(audio_data) == 0:
                    raise ValueError(f"Failed to load: {file_path}")

                if sample_rate is None:
                    sample_rate = sr
                elif sample_rate != sr:
                    raise ValueError(f"Sample rate mismatch: {file_path} ({sr} Hz vs {sample_rate} Hz)")

                signals.append((audio_data, file_path))

                if progress_callback:
                    progress_callback(i + 1, len(file_paths), f"Loaded {os.path.basename(file_path)}")

            except Exception as e:
                raise ValueError(f"Error loading {file_path}: {e}")

        # Process reference signal
        ref_signal, ref_path = signals[reference_index]
        ref_normalized = self._normalize_signal(ref_signal) if self.normalize else ref_signal.copy()
        ref_thresholded = self._apply_threshold(ref_normalized, self.threshold)

        # Determine max shift if not provided
        if max_shift is None:
            max_shift = len(ref_signal) // 2

        # Align all signals
        results = []

        for i, (signal, file_path) in enumerate(signals):
            if progress_callback:
                progress_callback(i + 1, len(signals), f"Aligning {os.path.basename(file_path)}...")

            # Process current signal
            normalized = self._normalize_signal(signal) if self.normalize else signal.copy()
            thresholded = self._apply_threshold(normalized, self.threshold)

            # Find optimal shift
            if i == reference_index:
                # Reference signal: no shift
                shift = 0
                correlation_peak = 1.0
            else:
                # Find shift using cross-correlation
                shift, correlation_peak = self._find_optimal_shift(
                    ref_thresholded,
                    thresholded,
                    max_shift
                )

            # Apply circular shift to original signal
            aligned_signal = self._circular_shift(signal, shift)

            results.append(AlignmentResult(
                original_signal=signal,
                aligned_signal=aligned_signal,
                shift_samples=shift,
                correlation_peak=correlation_peak,
                file_path=file_path,
                sample_rate=sample_rate
            ))

        if progress_callback:
            progress_callback(len(signals), len(signals), "Alignment complete!")

        return results

    def save_aligned_signals(
        self,
        results: List[AlignmentResult],
        output_dir: str,
        create_dir: bool = True
    ) -> List[str]:
        """
        Save aligned signals to WAV files.

        Args:
            results: List of AlignmentResult objects
            output_dir: Output directory path
            create_dir: Whether to create output directory if it doesn't exist

        Returns:
            List of output file paths
        """
        if create_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        output_paths = []

        for result in results:
            # Generate output filename
            basename = os.path.basename(result.file_path)
            output_path = os.path.join(output_dir, basename)

            # Save aligned signal
            self._save_wav_file(
                output_path,
                result.aligned_signal,
                result.sample_rate
            )

            output_paths.append(output_path)

        return output_paths

    def get_alignment_report(self, results: List[AlignmentResult]) -> Dict[str, Any]:
        """
        Generate alignment statistics report.

        Args:
            results: List of AlignmentResult objects

        Returns:
            Dictionary with alignment statistics
        """
        shifts = [r.shift_samples for r in results]
        correlations = [r.correlation_peak for r in results]
        sample_rate = results[0].sample_rate if results else 0

        # Convert shifts to milliseconds
        shifts_ms = [s / sample_rate * 1000 for s in shifts] if sample_rate > 0 else shifts

        return {
            "num_signals": len(results),
            "sample_rate": sample_rate,
            "shifts_samples": {
                "min": int(np.min(shifts)),
                "max": int(np.max(shifts)),
                "mean": float(np.mean(shifts)),
                "std": float(np.std(shifts))
            },
            "shifts_ms": {
                "min": float(np.min(shifts_ms)),
                "max": float(np.max(shifts_ms)),
                "mean": float(np.mean(shifts_ms)),
                "std": float(np.std(shifts_ms))
            },
            "correlations": {
                "min": float(np.min(correlations)),
                "max": float(np.max(correlations)),
                "mean": float(np.mean(correlations)),
                "std": float(np.std(correlations))
            },
            "reference_index": next((i for i, r in enumerate(results) if r.shift_samples == 0), 0)
        }

    # ========================================================================
    # Internal methods
    # ========================================================================

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to unit peak amplitude."""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val
        return signal.copy()

    def _apply_threshold(self, signal: np.ndarray, threshold: float) -> np.ndarray:
        """Apply threshold to remove noise and minor components."""
        # Normalize if not already
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            normalized = signal / max_val
        else:
            normalized = signal.copy()

        # Apply threshold
        thresholded = normalized.copy()
        thresholded[np.abs(thresholded) < threshold] = 0.0

        return thresholded

    def _find_optimal_shift(
        self,
        reference: np.ndarray,
        signal: np.ndarray,
        max_shift: int
    ) -> Tuple[int, float]:
        """
        Find optimal shift using cross-correlation.

        Args:
            reference: Reference signal (thresholded)
            signal: Signal to align (thresholded)
            max_shift: Maximum shift to search

        Returns:
            Tuple of (optimal_shift_samples, correlation_peak)
        """
        # Ensure signals are same length (pad if needed)
        N = max(len(reference), len(signal))

        if len(reference) < N:
            reference = np.pad(reference, (0, N - len(reference)), mode='constant')
        if len(signal) < N:
            signal = np.pad(signal, (0, N - len(signal)), mode='constant')

        # Compute cross-correlation using FFT (faster for large signals)
        correlation = np.correlate(reference, signal, mode='full')

        # Find peak within max_shift range
        center = len(correlation) // 2
        search_start = max(0, center - max_shift)
        search_end = min(len(correlation), center + max_shift + 1)

        search_region = correlation[search_start:search_end]
        peak_idx = np.argmax(np.abs(search_region))

        # Convert to shift value
        shift = peak_idx + search_start - center
        correlation_peak = float(search_region[peak_idx])

        # Normalize correlation
        ref_energy = np.sum(reference ** 2)
        sig_energy = np.sum(signal ** 2)
        if ref_energy > 0 and sig_energy > 0:
            correlation_peak /= np.sqrt(ref_energy * sig_energy)

        return shift, correlation_peak

    def _circular_shift(self, signal: np.ndarray, shift: int) -> np.ndarray:
        """
        Apply circular shift to signal.

        Positive shift moves signal forward (right), with beginning wrapping to end.
        Negative shift moves signal backward (left), with end wrapping to beginning.

        Args:
            signal: Input signal
            shift: Number of samples to shift (positive or negative)

        Returns:
            Circularly shifted signal
        """
        if shift == 0:
            return signal.copy()

        # Use numpy roll for circular shift
        # Note: np.roll with positive shift moves elements to the right
        return np.roll(signal, shift)

    def _load_wav_file(self, file_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load WAV file and return (audio_data, sample_rate)."""
        try:
            with wave.open(file_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                sample_width = wf.getsampwidth()

                # Read raw audio data
                raw_data = wf.readframes(n_frames)

                # Convert to numpy array
                if sample_width == 1:
                    dtype = np.uint8
                    audio_data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
                    audio_data = (audio_data - 128) / 128.0
                elif sample_width == 2:
                    dtype = np.int16
                    audio_data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
                    audio_data = audio_data / 32768.0
                elif sample_width == 4:
                    dtype = np.int32
                    audio_data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
                    audio_data = audio_data / 2147483648.0
                else:
                    return None, 0

                # Handle multi-channel audio (convert to mono)
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                elif n_channels > 2:
                    audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)

                return audio_data, sample_rate

        except Exception as e:
            return None, 0

    def _save_wav_file(self, file_path: str, audio_data: np.ndarray, sample_rate: int):
        """Save audio data to WAV file."""
        # Normalize and convert to int16
        if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0:
            normalized = audio_data / np.max(np.abs(audio_data)) * 0.95
        else:
            normalized = audio_data

        pcm = (np.clip(normalized, -1.0, 1.0) * 32767.0).astype(np.int16)

        # Write WAV file
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())


# ========================================================================
# Convenience functions
# ========================================================================

def align_impulse_responses(
    file_paths: List[str],
    output_dir: str,
    reference_index: int = 0,
    threshold: float = 0.3,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to align impulse responses.

    Args:
        file_paths: List of impulse response file paths
        output_dir: Output directory for aligned files
        reference_index: Index of reference signal
        threshold: Threshold for noise removal
        progress_callback: Optional progress callback

    Returns:
        Dictionary with alignment results and statistics
    """
    aligner = SignalAligner(threshold=threshold, normalize=True)

    # Align signals
    results = aligner.align_signals(
        file_paths,
        reference_index=reference_index,
        progress_callback=progress_callback
    )

    # Save aligned signals
    output_paths = aligner.save_aligned_signals(results, output_dir)

    # Generate report
    report = aligner.get_alignment_report(results)
    report['output_paths'] = output_paths
    report['output_dir'] = output_dir

    return report


def average_signals(
    file_paths: List[str],
    output_file: str,
    align_first: bool = True,
    reference_index: int = 0,
    threshold: float = 0.3,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Average multiple audio signals and save the result.

    Args:
        file_paths: List of audio file paths
        output_file: Output file path for averaged signal
        align_first: If True, align signals before averaging
        reference_index: Index of reference signal for alignment
        threshold: Threshold for noise removal during alignment
        progress_callback: Optional progress callback

    Returns:
        Dictionary with averaging results and statistics
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    if progress_callback:
        progress_callback(0, len(file_paths), "Loading signals...")

    # Load all signals
    signals = []
    sample_rate = None

    for i, file_path in enumerate(file_paths):
        aligner = SignalAligner(threshold=threshold, normalize=False)
        audio_data, sr = aligner._load_wav_file(file_path)

        if audio_data is None or len(audio_data) == 0:
            raise ValueError(f"Failed to load: {file_path}")

        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError(f"Sample rate mismatch: {file_path} ({sr} Hz vs {sample_rate} Hz)")

        signals.append(audio_data)

        if progress_callback:
            progress_callback(i + 1, len(file_paths), f"Loaded {os.path.basename(file_path)}")

    # Align signals if requested
    if align_first:
        if progress_callback:
            progress_callback(0, len(signals), "Aligning signals...")

        aligner = SignalAligner(threshold=threshold, normalize=True)
        results = aligner.align_signals(
            file_paths,
            reference_index=reference_index,
            progress_callback=progress_callback
        )

        # Use aligned signals
        signals = [r.aligned_signal for r in results]
        alignment_report = aligner.get_alignment_report(results)
    else:
        alignment_report = None

    # Find maximum length
    max_length = max(len(sig) for sig in signals)

    # Pad signals to same length
    padded_signals = []
    for sig in signals:
        if len(sig) < max_length:
            padded = np.pad(sig, (0, max_length - len(sig)), mode='constant')
            padded_signals.append(padded)
        else:
            padded_signals.append(sig)

    if progress_callback:
        progress_callback(1, 1, "Computing average...")

    # Stack and average
    stacked = np.stack(padded_signals, axis=0)
    averaged = np.mean(stacked, axis=0)

    # Save averaged signal
    if progress_callback:
        progress_callback(1, 1, "Saving averaged signal...")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    aligner = SignalAligner()
    aligner._save_wav_file(output_file, averaged, sample_rate)

    # Generate report
    report = {
        "num_signals": len(signals),
        "sample_rate": sample_rate,
        "output_file": output_file,
        "aligned": align_first,
        "averaged_length": len(averaged),
        "averaged_duration_s": len(averaged) / sample_rate,
        "averaged_rms": float(np.sqrt(np.mean(averaged ** 2))),
        "averaged_peak": float(np.max(np.abs(averaged))),
    }

    if alignment_report:
        report["alignment"] = alignment_report

    # Individual signal statistics
    report["individual_stats"] = {
        "lengths": {
            "min": int(np.min([len(s) for s in signals])),
            "max": int(np.max([len(s) for s in signals])),
            "mean": float(np.mean([len(s) for s in signals])),
        },
        "rms": {
            "min": float(np.min([np.sqrt(np.mean(s ** 2)) for s in signals])),
            "max": float(np.max([np.sqrt(np.mean(s ** 2)) for s in signals])),
            "mean": float(np.mean([np.sqrt(np.mean(s ** 2)) for s in signals])),
        },
        "peak": {
            "min": float(np.min([np.max(np.abs(s)) for s in signals])),
            "max": float(np.max([np.max(np.abs(s)) for s in signals])),
            "mean": float(np.mean([np.max(np.abs(s)) for s in signals])),
        },
    }

    if progress_callback:
        progress_callback(1, 1, "Complete!")

    return report


if __name__ == "__main__":
    # Example usage
    print("Signal Alignment Module")
    print("=" * 60)
    print("This module provides signal alignment functionality.")
    print("Import it into your application:")
    print()
    print("  from signal_alignment import SignalAligner")
    print("  aligner = SignalAligner(threshold=0.3)")
    print("  results = aligner.align_signals(file_paths)")
    print("  aligner.save_aligned_signals(results, output_dir)")
