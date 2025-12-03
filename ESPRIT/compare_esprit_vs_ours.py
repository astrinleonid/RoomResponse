"""
compare_esprit_vs_ours.py
Direct comparison: esprit.py vs our multi-band/multi-point implementation.

Runs both on the same 10-file subset for one frequency band.
"""
import sys
import numpy as np
from pathlib import Path
from glob import glob
import time

sys.path.insert(0, str(Path(__file__).parent))

# Our implementation
from preprocessing_minimal import (
    load_measurement_file,
    preprocess_measurement,
    MinimalPreprocessingConfig
)
from esprit_core import esprit_modal_identification
from band_processing import STANDARD_BANDS, process_band
from stabilization import ModeCandidate, stabilize_modes

# esprit.py functions (import directly)
from esprit import (
    build_hankel,
    TLS_ESPRIT_FromUs,
    LambdasToFZQ,
    read_index,
    load_cube,
    process_cube,
    band_presets
)


def run_our_implementation(measurement_files, fs=48000, freq_band_idx=0):
    """Run our multi-band + multi-point implementation."""

    print("="*80)
    print("OUR IMPLEMENTATION: Multi-Band + Multi-Point TLS-ESPRIT")
    print("="*80)

    # Import FrequencyBand to create custom band matching esprit.py
    from band_processing import FrequencyBand

    # Match esprit.py band_presets[freq_band_idx]
    esprit_presets = [
        (40.0, 500.0, 4, 0.01),   # band 0
        (500.0, 1000.0, 2, 0.005), # band 1
        (1000.0, 2000.0, 1, 0.002), # band 2
        (2000.0, 4000.0, 1, 0.001), # band 3
    ]

    f_min, f_max, decimation, exp_factor = esprit_presets[freq_band_idx]

    # Create custom band matching esprit.py exactly
    band = FrequencyBand(
        f_min=f_min,
        f_max=f_max,
        filter_order=6,  # Match esprit.py
        decimation=decimation,
        exp_factor=exp_factor,
        name=f"esprit_band_{freq_band_idx}"
    )

    print(f"\nBand (matching esprit.py band_presets[{freq_band_idx}]):")
    print(f"  Frequency range: {band.f_min}-{band.f_max} Hz")
    print(f"  Decimation: {band.decimation}x")
    print(f"  Exp factor: {band.exp_factor}")
    print()

    # Load and preprocess
    config = MinimalPreprocessingConfig(use_highpass=True, remove_contact=True)
    all_candidates = []

    print("Processing measurements:")
    for i, filepath in enumerate(measurement_files):
        try:
            filename = Path(filepath).stem
            print(f"  [{i+1}/{len(measurement_files)}] {filename}...", end=" ")

            # Load
            force, responses = load_measurement_file(filepath, skip_channel=2)

            # Preprocess
            processed, metadata = preprocess_measurement(force, responses, fs, config)

            # Band processing
            band_data, fs_band, _ = process_band(processed, fs, band, apply_preemphasis=True)

            # ESPRIT analysis
            result = esprit_modal_identification(
                band_data,
                fs=fs_band,
                model_order=30,
                freq_range=(band.f_min, band.f_max),
                use_tls=True,
                use_conjugate_pairing=True,
                use_multichannel=False,
                min_freq=band.f_min
            )

            print(f"{len(result.frequencies)} modes")

            # Convert to candidates
            for j in range(len(result.frequencies)):
                candidate = ModeCandidate(
                    frequency=result.frequencies[j],
                    damping=result.damping_ratios[j],
                    pole=result.poles[j],
                    mode_shape=result.mode_shapes[j] if result.mode_shapes is not None else None,
                    quality=1.0,
                    source_id=i,
                    band_name=band.name
                )
                all_candidates.append(candidate)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    print(f"\nTotal candidates: {len(all_candidates)}")

    # Stabilization
    stable_modes = stabilize_modes(
        all_candidates,
        freq_tol_hz=2.0,
        damping_tol=0.05,
        min_detections=3  # Must appear in >=3 measurements
    )

    print(f"Stable modes: {len(stable_modes)}\n")

    if len(stable_modes) > 0:
        print("Mode | Frequency (Hz) |  ±Std  | Damping (%) |  ±Std  | Detections")
        print("-----|----------------|--------|-------------|--------|------------")
        for i, mode in enumerate(stable_modes):
            print(f"{i:4d} | {mode.frequency:14.2f} | {mode.std_frequency:6.2f} | "
                  f"{mode.damping*100:11.2f} | {mode.std_damping*100:6.2f} | "
                  f"{mode.n_detections:10d}")

    return stable_modes


def run_esprit_py(index_path, cube_path, band_idx=0, K=30):
    """Run esprit.py implementation."""

    print("\n" + "="*80)
    print("ESPRIT.PY: Reference Implementation")
    print("="*80)

    # Load data
    R, M_raw, M_out, N_use, fs, names = read_index(index_path)
    y_cube = load_cube(cube_path, R, M_out, N_use)

    print(f"\nData loaded:")
    print(f"  R (excitation points): {R}")
    print(f"  M_out (channels): {M_out}")
    print(f"  N_use (samples): {N_use}")
    print(f"  fs: {fs} Hz")

    # Band processing
    current_preset = band_presets[band_idx]
    print(f"\nBand preset {band_idx}:")
    print(f"  Frequency range: {current_preset.low_freq}-{current_preset.high_freq} Hz")
    print(f"  Decimation: {current_preset.decimate_factor}x")
    print(f"  N_band: {current_preset.N_band}")

    processed_cube, fs_band = process_cube(y_cube, R, M_out, N_use, fs, current_preset, skip_m=2)
    N_band = processed_cube.shape[2]
    dt = 1.0 / fs_band

    print(f"  fs_band: {fs_band} Hz")
    print(f"  N_band (processed): {N_band}")

    # ESPRIT parameters
    L = N_band // 3
    M_eff = processed_cube.shape[1]

    print(f"\nESPRIT parameters:")
    print(f"  Window length L: {L}")
    print(f"  Model order K: {K}")
    print(f"  Effective channels: {M_eff}")

    # Process each excitation point
    all_modes_f = []
    all_modes_z = []
    all_modes_r = []

    print(f"\nProcessing {R} excitation points:")
    for r in range(R):
        print(f"  Point {r} ({names[r]})...", end=" ")

        # Build multi-channel Hankel
        big_H = np.vstack([
            build_hankel(processed_cube[r, me, :N_band], N_band, L)
            for me in range(M_eff)
        ])

        # SVD
        U, S, Vh = np.linalg.svd(big_H, full_matrices=False)
        Us = U[:, :K]

        # TLS-ESPRIT
        lam_re, lam_im = TLS_ESPRIT_FromUs(Us, M_eff * L, L, M_eff, K)

        # Convert to frequencies/damping
        F, Q, Z = LambdasToFZQ(lam_re, lam_im, K, dt)

        print(f"{len(F)} modes")

        for f, z in zip(F, Z):
            all_modes_f.append(f)
            all_modes_z.append(z)
            all_modes_r.append(r)

    print(f"\nTotal modes from all points: {len(all_modes_f)}")

    # Simple frequency clustering (matching esprit.py logic)
    if len(all_modes_f) > 0:
        from scipy.cluster.vq import kmeans, vq

        # Convert to array
        freqs_array = np.array(all_modes_f).reshape(-1, 1)

        # Estimate number of clusters (max 20)
        n_clusters = min(20, len(freqs_array) // 3)

        if n_clusters > 0:
            centroids, _ = kmeans(freqs_array, n_clusters)
            labels, _ = vq(freqs_array, centroids)

            # Extract stable modes (appear in >=3 points)
            stable_modes_esprit = []
            for cluster_id in range(n_clusters):
                cluster_mask = (labels == cluster_id)
                cluster_freqs = freqs_array[cluster_mask].flatten()
                cluster_zetas = np.array(all_modes_z)[cluster_mask]
                cluster_rs = np.array(all_modes_r)[cluster_mask]

                n_detections = len(cluster_freqs)

                if n_detections >= 3:
                    avg_f = np.mean(cluster_freqs)
                    std_f = np.std(cluster_freqs)
                    avg_z = np.mean(cluster_zetas)
                    std_z = np.std(cluster_zetas)

                    stable_modes_esprit.append({
                        'frequency': avg_f,
                        'std_freq': std_f,
                        'damping': avg_z,
                        'std_damp': std_z,
                        'n_detections': n_detections
                    })

            # Sort by frequency
            stable_modes_esprit = sorted(stable_modes_esprit, key=lambda x: x['frequency'])

            print(f"Stable modes (>=3 detections): {len(stable_modes_esprit)}\n")

            if len(stable_modes_esprit) > 0:
                print("Mode | Frequency (Hz) |  ±Std  | Damping (%) |  ±Std  | Detections")
                print("-----|----------------|--------|-------------|--------|------------")
                for i, mode in enumerate(stable_modes_esprit):
                    print(f"{i:4d} | {mode['frequency']:14.2f} | {mode['std_freq']:6.2f} | "
                          f"{mode['damping']*100:11.2f} | {mode['std_damp']*100:6.2f} | "
                          f"{mode['n_detections']:10d}")

            return stable_modes_esprit

    return []


def main():
    """Main comparison."""

    print("="*80)
    print("COMPARISON: esprit.py vs Our Implementation")
    print("="*80)
    print()

    # Parameters
    data_dir = "piano_point_responses"
    n_files = 10

    # Use SAME frequency range for both!
    # esprit.py band_presets[0]: 40-500 Hz, decimation=4, exp=0.01
    # Our STANDARD_BANDS[0]: 30-200 Hz, decimation=4, exp=0.3
    # Let's use band_presets[0] for esprit.py, and create matching band for ours
    freq_band_idx = 0  # esprit.py: 40-500 Hz

    # Find files
    measurement_files = sorted(glob(f"{data_dir}/*.txt"))[:n_files]

    print(f"Dataset: {len(measurement_files)} measurements from {data_dir}")
    print(f"Files: {', '.join([Path(f).stem for f in measurement_files])}")
    print(f"Frequency band index: {freq_band_idx}")
    print()

    # Run our implementation
    start = time.time()
    our_modes = run_our_implementation(measurement_files, freq_band_idx=freq_band_idx)
    our_time = time.time() - start

    # Run esprit.py
    index_path = "esprit_data/index.txt"
    cube_path = "esprit_data/y_cube.bin"

    start = time.time()
    esprit_modes = run_esprit_py(index_path, cube_path, band_idx=freq_band_idx, K=30)
    esprit_time = time.time() - start

    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nOur implementation:")
    print(f"  Stable modes: {len(our_modes)}")
    print(f"  Time: {our_time:.1f} seconds")

    print(f"\nesprit.py:")
    print(f"  Stable modes: {len(esprit_modes)}")
    print(f"  Time: {esprit_time:.1f} seconds")

    # Match modes between implementations
    if len(our_modes) > 0 and len(esprit_modes) > 0:
        print("\n" + "-"*80)
        print("Mode Matching (frequency tolerance: ±5 Hz)")
        print("-"*80)
        print("\nOur Freq (Hz) | esprit.py Freq (Hz) | Diff (Hz) | Our Damp (%) | esprit.py Damp (%)")
        print("--------------|---------------------|-----------|--------------|-------------------")

        matched = []
        for our_mode in our_modes:
            # Find closest esprit.py mode
            min_diff = float('inf')
            best_match = None

            for esp_mode in esprit_modes:
                diff = abs(our_mode.frequency - esp_mode['frequency'])
                if diff < min_diff:
                    min_diff = diff
                    best_match = esp_mode

            if best_match and min_diff < 5.0:  # Within 5 Hz
                print(f"{our_mode.frequency:13.2f} | {best_match['frequency']:19.2f} | "
                      f"{min_diff:7.2f} | {our_mode.damping*100:9.2f} | "
                      f"{best_match['damping']*100:14.2f}")
                matched.append(our_mode)

        print(f"\nMatched modes: {len(matched)}/{len(our_modes)}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
