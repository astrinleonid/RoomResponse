"""
convert_to_esprit_format.py
Convert piano_point_responses/*.txt files to esprit.py expected format.

esprit.py expects:
1. index.txt: Metadata file with format:
   # R=<num_points> M_raw=<channels> M_out=<channels> N_use=<samples> fs=<sample_rate>
   <point_name_1>
   <point_name_2>
   ...

2. y_cube.bin: Binary file with shape (R, M_out, N_use) as float64

This script converts individual .txt measurement files to this format.
"""
import sys
import numpy as np
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_minimal import load_measurement_file, preprocess_measurement, MinimalPreprocessingConfig


def convert_measurements_to_cube(data_dir: str, output_dir: str,
                                 pattern: str = "*.txt",
                                 max_files: int = None,
                                 skip_channel: int = 2,
                                 fs: float = 48000):
    """
    Convert individual measurement files to esprit.py format.

    Args:
        data_dir: Directory containing measurement files
        output_dir: Output directory for index.txt and y_cube.bin
        pattern: File pattern to match (default: *.txt)
        max_files: Maximum number of files to process (default: all)
        skip_channel: Channel to skip (default: 2)
        fs: Sampling frequency (default: 48000 Hz)
    """

    # Find measurement files
    search_path = Path(data_dir) / pattern
    measurement_files = sorted(glob(str(search_path)))

    if max_files is not None:
        measurement_files = measurement_files[:max_files]

    if len(measurement_files) == 0:
        print(f"ERROR: No files found matching {search_path}")
        sys.exit(1)

    print("="*80)
    print(f"Converting {len(measurement_files)} measurements to esprit.py format")
    print("="*80)
    print()

    # Load and preprocess all measurements
    config = MinimalPreprocessingConfig(use_highpass=True, remove_contact=True)

    all_processed = []
    valid_names = []

    for i, filepath in enumerate(measurement_files):
        try:
            filename = Path(filepath).stem  # Without extension
            print(f"[{i+1}/{len(measurement_files)}] Processing {filename}...", end=" ")

            # Load
            force, responses = load_measurement_file(filepath, skip_channel=skip_channel)

            # Preprocess
            processed, metadata = preprocess_measurement(force, responses, fs, config)

            print(f"OK ({processed.shape[0]} samples, {processed.shape[1]} channels)")

            all_processed.append(processed)
            valid_names.append(filename)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    if len(all_processed) == 0:
        print("\nERROR: No valid measurements loaded")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(all_processed)} measurements")

    # Determine cube dimensions
    R = len(all_processed)  # Number of excitation points
    M_out = all_processed[0].shape[1]  # Number of channels

    # Find minimum length (all measurements must have same length)
    min_length = min(data.shape[0] for data in all_processed)
    N_use = min_length

    print(f"\nCube dimensions:")
    print(f"  R (excitation points): {R}")
    print(f"  M_out (channels): {M_out}")
    print(f"  N_use (samples): {N_use}")
    print(f"  Total size: {R * M_out * N_use * 8 / 1024 / 1024:.2f} MB")

    # Create cube (R, M_out, N_use)
    # Note: esprit.py expects shape (R, M_out, N_use) in Fortran/column-major order when flattened
    y_cube = np.zeros((R, M_out, N_use), dtype=np.float64)

    for r in range(R):
        # Transpose to (M_out, N_use) and truncate to N_use
        y_cube[r, :, :] = all_processed[r][:N_use, :].T

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write index.txt
    index_path = output_path / "index.txt"
    print(f"\nWriting index file: {index_path}")

    with open(index_path, 'w') as f:
        # Header line
        f.write(f"# R={R} M_raw={M_out} M_out={M_out} N_use={N_use} fs={fs}\n")

        # Names (one per line)
        for name in valid_names:
            f.write(f"{name}\n")

    # Write y_cube.bin
    cube_path = output_path / "y_cube.bin"
    print(f"Writing binary cube: {cube_path}")

    # Flatten in C order (row-major) which matches esprit.py expectation
    flat = y_cube.flatten()
    flat.tofile(str(cube_path))

    print(f"\nConversion complete!")
    print(f"\nTo run esprit.py on this data:")
    print(f"  1. Edit esprit.py line 605:")
    print(f'     stabilization_diagram_with_shapes("{index_path.absolute()}", "{cube_path.absolute()}", band_index=0, K=30, selected_r="")')
    print(f"  2. Run: python esprit.py")
    print()
    print(f"Parameters:")
    print(f"  band_index: 0=Low(40-500Hz), 1=Mid-Low(500-1000Hz), 2=Mid-High(1000-2000Hz), 3=High(2000-4000Hz)")
    print(f"  K: Model order (try 20-40)")
    print(f'  selected_r: "" for all points, or "0,2,5" for specific points')
    print()

    return index_path, cube_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert measurements to esprit.py format')
    parser.add_argument('data_dir', help='Directory containing measurement .txt files')
    parser.add_argument('--output', '-o', default='esprit_data',
                       help='Output directory (default: esprit_data)')
    parser.add_argument('--max-files', '-n', type=int, default=None,
                       help='Maximum number of files to process (default: all)')
    parser.add_argument('--skip-channel', type=int, default=2,
                       help='Channel to skip (default: 2)')
    parser.add_argument('--fs', type=float, default=48000,
                       help='Sampling frequency in Hz (default: 48000)')

    args = parser.parse_args()

    convert_measurements_to_cube(
        args.data_dir,
        args.output,
        max_files=args.max_files,
        skip_channel=args.skip_channel,
        fs=args.fs
    )
