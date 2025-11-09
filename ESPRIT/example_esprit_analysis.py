"""
example_esprit_analysis.py
Example script demonstrating ESPRIT modal analysis on piano measurement data.

Usage:
    python example_esprit_analysis.py <measurement_file> [--gpu] [--order M] [--freq-range f_min f_max]
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from preprocessing import (
    load_measurement_file,
    preprocess_measurement,
    check_signal_quality,
    PreprocessingConfig
)
from esprit_core import (
    esprit_modal_identification,
    reconstruct_signal,
    ModalParameters
)


def plot_preprocessing_results(force: np.ndarray, responses: np.ndarray,
                               decay_segments: np.ndarray, metadata: dict,
                               fs: float, output_dir: Path):
    """Plot preprocessing diagnostic figures."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Time vectors
    t_full = np.arange(len(force)) / fs
    t_decay = np.arange(len(decay_segments)) / fs

    # Plot 1: Force signal with onset and contact markers
    ax = axes[0]
    ax.plot(t_full, force, 'b-', linewidth=0.5, label='Force')
    ax.axvline(metadata['onset_time_s'], color='g', linestyle='--', label='Onset')
    ax.axvline(metadata['contact_end_time_s'], color='r', linestyle='--', label='Contact end')
    ax.axvline(metadata['decay_start_index']/fs, color='orange', linestyle='--', label='Decay start')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Force Signal with Event Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Raw response signals (all channels)
    ax = axes[1]
    for ch in range(responses.shape[1]):
        ax.plot(t_full, responses[:, ch], linewidth=0.5, alpha=0.7, label=f'Ch {ch}')
    ax.axvline(metadata['contact_end_time_s'], color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response')
    ax.set_title('Raw Response Signals (All Channels)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Windowed decay segments
    ax = axes[2]
    for ch in range(decay_segments.shape[1]):
        ax.plot(t_decay, decay_segments[:, ch], linewidth=0.5, alpha=0.7, label=f'Ch {ch}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Windowed Response')
    ax.set_title('Exponentially Windowed Decay Segments')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'preprocessing.png', dpi=150)
    plt.close()


def plot_modal_results(modal_params: ModalParameters, decay_segments: np.ndarray,
                       fs: float, output_dir: Path):
    """Plot modal identification results."""
    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Singular values
    ax1 = plt.subplot(3, 2, 1)
    ax1.semilogy(modal_params.singular_values[:50], 'bo-', markersize=4)
    ax1.axvline(modal_params.model_order, color='r', linestyle='--', label=f'Order M={modal_params.model_order}')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Value Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Identified poles in s-plane
    ax2 = plt.subplot(3, 2, 2)
    poles = modal_params.poles
    ax2.plot(poles.real, poles.imag, 'ro', markersize=8, label='Poles')
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.axvline(0, color='k', linewidth=0.5)
    ax2.set_xlabel('Real (σ) [rad/s]')
    ax2.set_ylabel('Imag (ω) [rad/s]')
    ax2.set_title('Poles in Complex Plane')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Natural frequencies
    ax3 = plt.subplot(3, 2, 3)
    ax3.stem(modal_params.frequencies, np.ones_like(modal_params.frequencies),
             basefmt=' ', linefmt='b-', markerfmt='bo')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title(f'Identified Natural Frequencies ({len(modal_params.frequencies)} modes)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Damping ratios
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(modal_params.frequencies, modal_params.damping_ratios * 100, 'go', markersize=8)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Damping Ratio (%)')
    ax4.set_title('Modal Damping Ratios')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Mode shape magnitudes
    ax5 = plt.subplot(3, 2, 5)
    n_modes = min(8, len(modal_params.frequencies))  # Plot first 8 modes
    mode_mags = np.abs(modal_params.mode_shapes[:n_modes, :])
    im = ax5.imshow(mode_mags, aspect='auto', cmap='viridis', interpolation='nearest')
    ax5.set_xlabel('Channel')
    ax5.set_ylabel('Mode')
    ax5.set_title('Mode Shape Magnitudes (first 8 modes)')
    plt.colorbar(im, ax=ax5)

    # Plot 6: Mode shape phases
    ax6 = plt.subplot(3, 2, 6)
    mode_phases = np.angle(modal_params.mode_shapes[:n_modes, :], deg=True)
    im = ax6.imshow(mode_phases, aspect='auto', cmap='hsv', interpolation='nearest',
                    vmin=-180, vmax=180)
    ax6.set_xlabel('Channel')
    ax6.set_ylabel('Mode')
    ax6.set_title('Mode Shape Phases (degrees)')
    plt.colorbar(im, ax=ax6, label='Phase (deg)')

    plt.tight_layout()
    plt.savefig(output_dir / 'modal_results.png', dpi=150)
    plt.close()


def plot_reconstruction(decay_segments: np.ndarray, modal_params: ModalParameters,
                       fs: float, output_dir: Path):
    """Plot signal reconstruction comparison."""
    dt = 1.0 / fs
    n_samples = len(decay_segments)

    # Reconstruct signal
    reconstructed = reconstruct_signal(modal_params.mode_shapes, modal_params.poles,
                                      n_samples, dt)

    # Time vector
    t = np.arange(n_samples) / fs

    # Plot for each channel
    n_channels = decay_segments.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3*n_channels))

    if n_channels == 1:
        axes = [axes]

    for ch in range(n_channels):
        ax = axes[ch]

        # Plot measured and reconstructed
        ax.plot(t, decay_segments[:, ch], 'b-', linewidth=1, alpha=0.7, label='Measured')
        ax.plot(t, reconstructed[:, ch], 'r--', linewidth=1, label='Reconstructed')

        # Compute error
        error = decay_segments[:, ch] - reconstructed[:, ch]
        rms_error = np.sqrt(np.mean(error**2))
        rms_signal = np.sqrt(np.mean(decay_segments[:, ch]**2))
        error_db = 20 * np.log10(rms_error / rms_signal)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Channel {ch}')
        ax.set_title(f'Channel {ch} Reconstruction (Error: {error_db:.1f} dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction.png', dpi=150)
    plt.close()


def save_results(modal_params: ModalParameters, metadata: dict,
                output_dir: Path, filename_stem: str):
    """Save modal parameters to files."""

    # Save modal table (CSV)
    modal_table = []
    for k in range(len(modal_params.poles)):
        row = {
            'mode_id': k,
            'f_hz': modal_params.frequencies[k],
            'zeta': modal_params.damping_ratios[k],
            'sigma': modal_params.poles[k].real,
            'omega': modal_params.poles[k].imag,
            'pole_real': modal_params.poles[k].real,
            'pole_imag': modal_params.poles[k].imag,
        }
        modal_table.append(row)

    import pandas as pd
    df = pd.DataFrame(modal_table)
    df.to_csv(output_dir / f'{filename_stem}_modal_table.csv', index=False)

    # Save mode shapes (one file per mode)
    n_channels = modal_params.mode_shapes.shape[1]
    for k in range(len(modal_params.poles)):
        shape_data = []
        for ch in range(n_channels):
            phi = modal_params.mode_shapes[k, ch]
            shape_data.append({
                'sensor_id': ch,
                'phi_real': phi.real,
                'phi_imag': phi.imag,
                'phi_mag': np.abs(phi),
                'phi_phase_deg': np.angle(phi, deg=True)
            })
        df_shape = pd.DataFrame(shape_data)
        df_shape.to_csv(output_dir / f'{filename_stem}_mode_{k:03d}_shape.csv', index=False)

    # Save NPZ bundle
    np.savez(
        output_dir / f'{filename_stem}_modes.npz',
        f_hz=modal_params.frequencies,
        zeta=modal_params.damping_ratios,
        poles=modal_params.poles,
        phi=modal_params.mode_shapes,
        model_order=modal_params.model_order,
        singular_values=modal_params.singular_values,
        metadata=metadata
    )

    # Save summary JSON
    summary = {
        'n_modes': len(modal_params.poles),
        'model_order': modal_params.model_order,
        'frequency_range_hz': [float(modal_params.frequencies.min()),
                               float(modal_params.frequencies.max())],
        'damping_range': [float(modal_params.damping_ratios.min()),
                         float(modal_params.damping_ratios.max())],
        'preprocessing': metadata
    }

    with open(output_dir / f'{filename_stem}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - Modal table: {filename_stem}_modal_table.csv")
    print(f"  - Mode shapes: {filename_stem}_mode_XXX_shape.csv")
    print(f"  - NPZ bundle: {filename_stem}_modes.npz")
    print(f"  - Summary: {filename_stem}_summary.json")


def main():
    parser = argparse.ArgumentParser(description='ESPRIT modal analysis on piano measurement')
    parser.add_argument('measurement_file', type=str, help='Path to measurement TSV file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--order', '-M', type=int, default=20, help='Model order (default: 20)')
    parser.add_argument('--freq-range', nargs=2, type=float, default=[0, 1000],
                       help='Frequency range [min max] in Hz (default: 0 1000)')
    parser.add_argument('--fs', type=float, default=48000, help='Sampling frequency in Hz')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory (default: same as input file)')
    parser.add_argument('--window-length-frac', type=float, default=0.5,
                       help='Hankel window length as fraction of decay (default: 0.5)')

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.measurement_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / 'esprit_results'
    output_dir.mkdir(exist_ok=True, parents=True)

    filename_stem = input_path.stem

    print("=" * 70)
    print("ESPRIT Modal Identification")
    print("=" * 70)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model order: {args.order}")
    print(f"Frequency range: {args.freq_range[0]:.1f} - {args.freq_range[1]:.1f} Hz")
    print(f"GPU acceleration: {'Enabled' if args.gpu else 'Disabled'}")
    print()

    # Load measurement data
    print("Loading measurement data...")
    force, responses = load_measurement_file(str(input_path), skip_channel=2)
    print(f"  Force shape: {force.shape}")
    print(f"  Responses shape: {responses.shape}")
    print(f"  Sampling frequency: {args.fs} Hz")
    print(f"  Duration: {len(force)/args.fs:.3f} s")

    # Check signal quality
    print("\nChecking signal quality...")
    quality = check_signal_quality(force, responses)
    print(f"  Force clipped: {quality['force_clipped']}")
    print(f"  Double hit detected: {quality['double_hit']}")
    print(f"  SNR (dB): {quality['snr_db']}")

    # Preprocessing
    print("\nPreprocessing...")
    config = PreprocessingConfig()
    decay_segments, metadata = preprocess_measurement(force, responses, args.fs, config)
    print(f"  Onset time: {metadata['onset_time_s']:.4f} s")
    print(f"  Contact end: {metadata['contact_end_time_s']:.4f} s")
    print(f"  Decay duration: {metadata['decay_duration_s']:.4f} s")
    print(f"  Decay samples: {metadata['decay_length']}")

    # Plot preprocessing
    print("\nGenerating preprocessing plots...")
    plot_preprocessing_results(force, responses, decay_segments, metadata, args.fs, output_dir)

    # Modal identification
    print("\nRunning ESPRIT modal identification...")
    window_length = int(len(decay_segments) * args.window_length_frac)
    print(f"  Hankel window length: {window_length} ({args.window_length_frac*100:.0f}% of decay)")

    modal_params = esprit_modal_identification(
        decay_segments,
        fs=args.fs,
        model_order=args.order,
        window_length=window_length,
        use_gpu=args.gpu,
        max_damping=0.2,
        freq_range=tuple(args.freq_range),
        ref_sensor=0
    )

    print(f"\n{modal_params}")
    print(f"\nIdentified {len(modal_params.frequencies)} modes:")
    for k, (f, zeta) in enumerate(zip(modal_params.frequencies, modal_params.damping_ratios)):
        print(f"  Mode {k}: f = {f:8.3f} Hz, zeta = {zeta*100:5.2f} %")

    # Plot results
    print("\nGenerating result plots...")
    plot_modal_results(modal_params, decay_segments, args.fs, output_dir)
    plot_reconstruction(decay_segments, modal_params, args.fs, output_dir)

    # Save results
    print("\nSaving results...")
    save_results(modal_params, metadata, output_dir, filename_stem)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
