#!/usr/bin/env python3
"""
Calibrate Take2 Unique Channels using Take1 Mode References

This script:
1. Loads Take1 aggregated ESPRIT results (mode frequencies, damping, amplitudes for 5 channels)
2. Fits Take2 signals to Take1's mode parameters using least squares
3. Calibrates using common channel comparison
4. Exports combined results with 7 channels (Take1 channels 0-4 + Take2 channels 5-6)

Channel Mapping:
    Take1 recording → Take1 analysis    Take2 recording → Take2 analysis    Status
    ch2             → 0                 ch1             → 0                 Common
    ch3             → 1                 ch2             → 1                 Common
    ch4             → 2                 ch3             → 2                 Common
    ch5             → 3                 ch4             → 3                 Common
    ch6             → 4                 ch5             → 4                 Common
    -               → -                 ch6             → 5                 UNIQUE
    -               → -                 ch7             → 6                 UNIQUE
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.signal import butter, filtfilt


# Band definitions (matching ESPRIT processing)
BANDS = [
    {"name": "40_500", "low": 40, "high": 500},
    {"name": "500_1000", "low": 500, "high": 1000},
    {"name": "1000_2000", "low": 1000, "high": 2000},
    {"name": "2000_4000", "low": 2000, "high": 4000},
]

SAMPLE_RATE = 48000


def extract_scenario_num(name: str) -> int:
    """Extract scenario number from directory name like 'Ivers&Pond-Scenario42-Take1'"""
    match = re.search(r'Scenario(\d+)', name)
    return int(match.group(1)) if match else 0


def load_take1_results(band_name: str) -> Optional[Dict[str, Any]]:
    """Load Take1 aggregated results for a specific band."""
    results_file = Path(f"ESPRIT/Ivers_Pond_Take1/results_{band_name}.json")
    if not results_file.exists():
        print(f"ERROR: Take1 results not found: {results_file}")
        return None

    with open(results_file) as f:
        return json.load(f)


def load_take2_averaged_response(scenario_dir: Path, channels: List[int]) -> Optional[np.ndarray]:
    """
    Load averaged response for specified channels from Take2 scenario.

    Tries two methods:
    1. Pre-computed averaged_responses/*.npy files
    2. Compute average from impulse_responses/*.npy files

    Args:
        scenario_dir: Path to scenario directory
        channels: List of channel indices to load (1-7 for Take2 recording channels)

    Returns:
        Array of shape (num_channels, num_samples) or None if loading fails
    """
    # Method 1: Try pre-computed averaged responses
    avg_dir = scenario_dir / "averaged_responses"
    if avg_dir.exists():
        signals = []
        all_found = True
        for ch in channels:
            avg_file = avg_dir / f"average_ch{ch}.npy"
            if avg_file.exists():
                signals.append(np.load(avg_file))
            else:
                all_found = False
                break

        if all_found:
            return np.vstack(signals)

    # Method 2: Compute average from impulse responses
    impulse_dir = scenario_dir / "impulse_responses"
    if not impulse_dir.exists():
        return None

    # Group impulse files by channel
    # Files are like: impulse_001_ch1.npy, impulse_001_ch2.npy, etc.
    channel_files = {ch: [] for ch in channels}

    for f in sorted(impulse_dir.glob("impulse_*_ch*.npy")):
        # Extract channel from filename: impulse_XXX_chN.npy
        parts = f.stem.split("_ch")
        if len(parts) == 2:
            try:
                ch = int(parts[1])
                if ch in channel_files:
                    channel_files[ch].append(f)
            except ValueError:
                continue

    # Average each channel
    averaged_signals = []
    for ch in channels:
        files = channel_files[ch]
        if not files:
            return None  # Missing channel data

        signals = []
        for f in files:
            try:
                sig = np.load(f)
                signals.append(sig)
            except Exception:
                continue

        if not signals:
            return None

        # Find minimum length and truncate
        min_len = min(len(s) for s in signals)
        truncated = [s[:min_len] for s in signals]

        # Average
        avg_signal = np.mean(truncated, axis=0)
        averaged_signals.append(avg_signal)

    if len(averaged_signals) != len(channels):
        return None

    return np.vstack(averaged_signals)


def bandpass_filter(signal: np.ndarray, low_freq: float, high_freq: float,
                    fs: int = SAMPLE_RATE, order: int = 4) -> np.ndarray:
    """Apply bandpass filter to signal."""
    nyq = fs / 2
    low = low_freq / nyq
    high = high_freq / nyq

    # Ensure frequencies are in valid range
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.999))

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=-1)


def fit_to_reference_modes(
    signal: np.ndarray,
    freqs: List[float],
    zetas: List[float],
    fs: int = SAMPLE_RATE,
    band_low: float = 40,
    band_high: float = 500
) -> List[List[Dict[str, float]]]:
    """
    Fit signal to reference mode frequencies/damping using least squares.

    For each mode (f, zeta), constructs damped sinusoid basis:
        y(t) = a * exp(-alpha * t) * cos(omega * t) + b * exp(-alpha * t) * sin(omega * t)

    Solves for (a, b) per channel using least squares.

    Args:
        signal: Array of shape (num_channels, num_samples)
        freqs: List of mode frequencies (Hz)
        zetas: List of damping ratios
        fs: Sample rate
        band_low: Low frequency for bandpass filter
        band_high: High frequency for bandpass filter

    Returns:
        List of [num_modes][num_channels] with {"real": a, "imag": b} dicts
    """
    num_channels, num_samples = signal.shape
    t = np.arange(num_samples) / fs

    # Apply bandpass filter to signal
    filtered_signal = bandpass_filter(signal, band_low, band_high, fs)

    amplitudes = []

    for f, zeta in zip(freqs, zetas):
        omega = 2 * np.pi * f
        alpha = zeta * omega

        # Basis functions (damped sinusoids)
        decay = np.exp(-alpha * t)
        cos_basis = decay * np.cos(omega * t)
        sin_basis = decay * np.sin(omega * t)
        basis_matrix = np.vstack([cos_basis, sin_basis]).T  # (N, 2)

        mode_amps = []
        for ch in range(num_channels):
            # Least squares: y = a*cos_basis + b*sin_basis
            coeffs, _, _, _ = np.linalg.lstsq(basis_matrix, filtered_signal[ch], rcond=None)
            a, b = coeffs
            mode_amps.append({"real": float(a), "imag": float(b)})

        amplitudes.append(mode_amps)

    return amplitudes


def compute_calibration_factors(
    take1_amps: List[List[Dict[str, float]]],
    take2_amps: List[List[Dict[str, float]]],
    num_common_channels: int = 5,
    amplitude_threshold: float = 1e-8
) -> Tuple[List[float], List[float]]:
    """
    Compute calibration factors by comparing Take1 and Take2 amplitudes on common channels.

    Args:
        take1_amps: Take1 amplitudes [num_modes][num_channels]
        take2_amps: Take2 amplitudes [num_modes][num_channels]
        num_common_channels: Number of common channels (0-4)
        amplitude_threshold: Minimum amplitude to include in ratio computation

    Returns:
        (scale_factors, phase_offsets) for each common channel
    """
    scale_factors = []
    phase_offsets = []

    for ch in range(num_common_channels):
        ratios = []
        phases = []

        for mode_idx in range(len(take1_amps)):
            t1 = take1_amps[mode_idx][ch]
            t2 = take2_amps[mode_idx][ch]

            amp1 = complex(t1["real"], t1["imag"])
            amp2 = complex(t2["real"], t2["imag"])

            if abs(amp2) > amplitude_threshold and abs(amp1) > amplitude_threshold:
                ratios.append(abs(amp1) / abs(amp2))
                phases.append(np.angle(amp1) - np.angle(amp2))

        if ratios:
            scale_factors.append(np.median(ratios))
            phase_offsets.append(np.median(phases))
        else:
            scale_factors.append(1.0)
            phase_offsets.append(0.0)

    return scale_factors, phase_offsets


def apply_calibration(
    amplitudes: List[Dict[str, float]],
    scale: float,
    phase: float
) -> List[Dict[str, float]]:
    """Apply calibration (scale and phase) to amplitude list."""
    calibrated = []
    for amp in amplitudes:
        c = complex(amp["real"], amp["imag"])
        c_cal = c * scale * np.exp(1j * phase)
        calibrated.append({"real": float(c_cal.real), "imag": float(c_cal.imag)})
    return calibrated


def find_scenarios() -> Tuple[Dict[int, Path], Dict[int, Path], set, set]:
    """
    Find all Take1 and Take2 scenarios.

    Returns:
        (take1_dirs, take2_dirs, take1_nums, take2_nums)
    """
    piano_dir = Path("piano")

    take1_dirs = {}
    take2_dirs = {}

    for d in piano_dir.glob("Ivers&Pond-Scenario*-Take1"):
        num = extract_scenario_num(d.name)
        take1_dirs[num] = d

    for d in piano_dir.glob("Ivers&Pond-Scenario*-Take2"):
        num = extract_scenario_num(d.name)
        take2_dirs[num] = d

    take1_nums = set(take1_dirs.keys())
    take2_nums = set(take2_dirs.keys())

    return take1_dirs, take2_dirs, take1_nums, take2_nums


def process_band(band_idx: int) -> Optional[Dict[str, Any]]:
    """
    Process a single frequency band.

    Args:
        band_idx: Band index (0-3)

    Returns:
        Combined results dict or None on failure
    """
    band = BANDS[band_idx]
    band_name = band["name"]
    band_low = band["low"]
    band_high = band["high"]

    print(f"\n{'='*70}")
    print(f"Processing Band {band_idx}: {band_low}-{band_high} Hz")
    print(f"{'='*70}")

    # Load Take1 results
    take1_results = load_take1_results(band_name)
    if take1_results is None:
        return None

    common_f = take1_results["common_f"]
    common_z = take1_results["common_z"]
    take1_amplitudes = take1_results["amplitudes_in_m"]
    take1_names = take1_results["names"]

    print(f"Take1: {len(common_f)} modes, {len(take1_names)} scenarios")

    # Find scenarios
    take1_dirs, take2_dirs, take1_nums, take2_nums = find_scenarios()
    common_nums = take1_nums & take2_nums

    print(f"Take1 scenarios: {len(take1_nums)}")
    print(f"Take2 scenarios: {len(take2_nums)}")
    print(f"Overlapping: {len(common_nums)}")

    # Take2 channels to load: 1-7 (recording channels) -> analysis indices 0-6
    # For fitting, we need all 7 channels
    take2_recording_channels = [1, 2, 3, 4, 5, 6, 7]

    # Fit Take2 signals to Take1 modes for overlapping scenarios
    print("\nFitting Take2 signals to Take1 mode frequencies...")

    take2_fitted_amps = {}  # {scenario_num: fitted_amplitudes}

    for scenario_num in sorted(common_nums):
        scenario_dir = take2_dirs[scenario_num]

        # Load averaged response
        signal = load_take2_averaged_response(scenario_dir, take2_recording_channels)
        if signal is None:
            print(f"  Scenario {scenario_num}: SKIP (missing data)")
            continue

        # Fit to Take1 modes
        fitted = fit_to_reference_modes(
            signal, common_f, common_z, SAMPLE_RATE, band_low, band_high
        )
        take2_fitted_amps[scenario_num] = fitted

        print(f"  Scenario {scenario_num}: OK ({len(fitted)} modes fitted)")

    if not take2_fitted_amps:
        print("ERROR: No Take2 scenarios could be fitted")
        return None

    # Compute calibration factors using common channels (indices 0-4)
    print("\nComputing calibration factors from common channels...")

    # Map scenario names to indices in Take1 results
    name_to_idx = {name: idx for idx, name in enumerate(take1_names)}

    # Collect calibration data from all overlapping scenarios
    all_scale_factors = [[] for _ in range(5)]
    all_phase_offsets = [[] for _ in range(5)]

    for scenario_num, fitted in take2_fitted_amps.items():
        # Find corresponding Take1 scenario
        # Take1 names are like "00", "01", etc. (0-based from scenario numbers)
        take1_name = f"{scenario_num - 1:02d}"

        if take1_name not in name_to_idx:
            # Try alternate numbering (89 was renumbered to 88, so 88->87)
            if scenario_num == 89:
                take1_name = "87"
            if take1_name not in name_to_idx:
                continue

        take1_idx = name_to_idx[take1_name]

        # Compare amplitudes for common channels
        for mode_idx in range(len(common_f)):
            for ch in range(5):  # Common channels 0-4
                t1_amp = take1_amplitudes[mode_idx][ch]
                t2_amp = fitted[mode_idx][ch]

                amp1 = complex(t1_amp["real"], t1_amp["imag"])
                amp2 = complex(t2_amp["real"], t2_amp["imag"])

                if abs(amp2) > 1e-10 and abs(amp1) > 1e-10:
                    all_scale_factors[ch].append(abs(amp1) / abs(amp2))
                    all_phase_offsets[ch].append(np.angle(amp1) - np.angle(amp2))

    # Compute median calibration factors per channel
    scale_factors = []
    phase_offsets = []

    for ch in range(5):
        if all_scale_factors[ch]:
            scale_factors.append(np.median(all_scale_factors[ch]))
            phase_offsets.append(np.median(all_phase_offsets[ch]))
            print(f"  Channel {ch}: scale={scale_factors[-1]:.4f}, phase={np.degrees(phase_offsets[-1]):.1f}°")
        else:
            scale_factors.append(1.0)
            phase_offsets.append(0.0)
            print(f"  Channel {ch}: NO DATA (using defaults)")

    # Average scale factor for unique channels
    avg_scale = np.mean(scale_factors)
    avg_phase = np.mean(phase_offsets)
    print(f"\n  Average for unique channels: scale={avg_scale:.4f}, phase={np.degrees(avg_phase):.1f}°")

    # Build combined amplitudes with correct channel mapping:
    # Output channels: 0, 1, 2(calibration=zeros), 3, 4, 5, 6, 7
    #
    # Take1 has 5 analysis channels (0-4) with actual mic data:
    #   Take1 ch0 -> Output ch0
    #   Take1 ch1 -> Output ch1
    #   (none)    -> Output ch2 (calibration = zeros)
    #   Take1 ch2 -> Output ch3
    #   Take1 ch3 -> Output ch4
    #   Take1 ch4 -> Output ch5
    #
    # Take2 unique channels (only the 2 NEW physical mics):
    #   Take2 analysis ch5 (rec ch6) -> Output ch6 (unique)
    #   Take2 analysis ch6 (rec ch7) -> Output ch7 (unique)
    #
    # Note: Take2 analysis channels 0-4 are SAME mics as Take1, so we don't use them.
    # Only Take2 channels 5,6 provide new data from unique physical microphones.

    print("\nBuilding combined amplitudes (8 channels)...")
    print("  ch0,1: From Take1 ch0,1")
    print("  ch2: calibration (zeros)")
    print("  ch3,4,5: From Take1 ch2,3,4")
    print("  ch6,7: From Take2 unique mics")

    combined_amplitudes = []

    for mode_idx in range(len(common_f)):
        mode_amps = [{"real": 0.0, "imag": 0.0} for _ in range(8)]  # 8 output channels

        # Map Take1 channels with calibration gap at output ch2:
        # Take1 ch0 -> out ch0, Take1 ch1 -> out ch1
        # Take1 ch2 -> out ch3, Take1 ch3 -> out ch4, Take1 ch4 -> out ch5
        take1_to_output = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5}
        for take1_ch, out_ch in take1_to_output.items():
            if take1_ch < len(take1_amplitudes[mode_idx]):
                mode_amps[out_ch] = take1_amplitudes[mode_idx][take1_ch]

        # Output ch2 stays zeros (calibration)

        # Take2 unique channels (analysis indices 5,6 -> output ch6,7)
        # These are the ONLY new mics from Take2 (recording ch6, ch7)
        for take2_ch, out_ch in [(5, 6), (6, 7)]:
            # Average amplitudes from all fitted Take2 scenarios
            real_sum = 0.0
            imag_sum = 0.0
            count = 0

            for scenario_num, fitted in take2_fitted_amps.items():
                amp = fitted[mode_idx][take2_ch]
                real_sum += amp["real"]
                imag_sum += amp["imag"]
                count += 1

            if count > 0:
                avg_real = real_sum / count
                avg_imag = imag_sum / count

                # Apply calibration
                c = complex(avg_real, avg_imag) * avg_scale * np.exp(1j * avg_phase)
                mode_amps[out_ch] = {"real": float(c.real), "imag": float(c.imag)}

        combined_amplitudes.append(mode_amps)

    # Build output
    # Note: selected_r_indices should be RECEIVER indices [0,1,2,3,4,5,6,7] for 8 channels,
    # NOT scenario indices. The merge script uses this to map amplitudes to decka_coeff columns.
    output = {
        "common_f": common_f,
        "common_z": common_z,
        "signed_shapes": take1_results.get("signed_shapes", []),
        "participation": take1_results.get("participation", []),
        "amplitudes_in_m": combined_amplitudes,
        "names": take1_names,
        "selected_r_indices": list(range(8)),  # 8 receiver channels: 0-7
        "calibration_info": {
            "scale_factors": scale_factors,
            "phase_offsets_deg": [np.degrees(p) for p in phase_offsets],
            "avg_scale": avg_scale,
            "avg_phase_deg": np.degrees(avg_phase),
            "num_overlapping_scenarios": len(take2_fitted_amps),
        }
    }

    print(f"\nOK Band {band_idx}: {len(common_f)} modes with 8 channels (5 Take1 + calibration + 2 Take2 unique)")

    return output


def main():
    print("="*70)
    print("Take2 Channel Calibration and Integration")
    print("="*70)
    print()
    print("Goal: Derive mode coefficients for Take2 unique channels (5, 6)")
    print("      using Take1 mode frequencies as reference")
    print()

    output_dir = Path("ESPRIT/Ivers_Pond_Combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for band_idx in range(4):
        result = process_band(band_idx)

        if result is not None:
            band = BANDS[band_idx]
            output_file = output_dir / f"results_{band['name']}.json"

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"  Saved: {output_file}")
            success_count += 1

    print()
    print("="*70)
    print(f"Complete: {success_count}/4 bands processed")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
