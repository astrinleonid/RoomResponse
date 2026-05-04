"""
Example usage of esprit_streaming.py for real-time processing.

This demonstrates how to process measurements as they arrive,
which allows data collection and ESPRIT calculation to progress in parallel.
"""

import numpy as np
from esprit_streaming import StreamingESPRITProcessor
from esprit import read_index, load_cube


def example_with_existing_data():
    """
    Example: Process existing cube data one-by-one (simulating streaming).
    """
    # Load data
    index_path = r"esprit_data\index.txt"
    cube_path = r"esprit_data\y_cube.bin"

    R, M_raw, M_out, N_use, fs, names = read_index(index_path)
    y_cube = load_cube(cube_path, R, M_out, N_use)

    print(f"Loaded data cube: {R} points, {M_out} channels, {N_use} samples @ {fs} Hz")
    print(f"Point names: {names}\n")

    # Initialize processor
    processor = StreamingESPRITProcessor(
        M_out=M_out,
        N_use=N_use,
        fs=fs,
        band_index=0,  # 40-500 Hz band
        L_fraction=0.5,
        K=30,
        skip_m=2  # Skip calibration channel
    )

    # Process measurements one-by-one
    print("\n" + "="*70)
    print("STREAMING PROCESSING (one measurement at a time)")
    print("="*70)

    individual_results = []

    for r in range(R):
        # Simulate getting measurement as it arrives
        y_raw = y_cube[r, :, :]  # Shape: (M_out, N_use)

        # Process immediately
        result = processor.process_measurement(r, y_raw)
        individual_results.append(result)

        # You can examine results immediately
        print(f"\nPoint {r} ({names[r]}): {len(result['frequencies'])} modes found")
        if len(result['frequencies']) > 0:
            print(f"  Frequencies: {result['frequencies'][:5]}...")  # First 5

    # Optionally show stabilization diagram during processing
    print("\n" + "="*70)
    print("STABILIZATION DIAGRAM (current progress)")
    print("="*70)
    processor.get_current_stabilization_plot()

    # Get final aggregated results
    print("\n" + "="*70)
    print("FINAL AGGREGATION")
    print("="*70)

    final_results = processor.get_final_results(
        names=names,
        output_file="streaming_results.json"
    )

    # Show mode shapes
    processor.plot_mode_shapes(final_results)

    return processor, individual_results, final_results


def example_real_time_simulation():
    """
    Example: Simulating real-time acquisition where measurements arrive one-by-one.
    """
    import time

    # Simulate measurement parameters
    M_out = 6  # 5 sensors + 1 force
    N_use = 28800  # 600ms @ 48kHz
    fs = 48000

    # Initialize processor
    processor = StreamingESPRITProcessor(
        M_out=M_out,
        N_use=N_use,
        fs=fs,
        band_index=0,
        L_fraction=0.5,
        K=30,
        skip_m=2
    )

    print("\n" + "="*70)
    print("REAL-TIME SIMULATION")
    print("="*70)
    print("Simulating measurements arriving one-by-one...")

    # Simulate 10 measurement points
    for r in range(10):
        # Simulate waiting for measurement to complete
        print(f"\n[t={r*2:.1f}s] Acquiring measurement {r}...", end=" ")
        time.sleep(0.1)  # Simulated acquisition time

        # Generate simulated measurement (normally this would come from hardware)
        y_raw = np.random.randn(M_out, N_use) * 0.1

        # Add some synthetic decaying sinusoids
        t = np.arange(N_use) / fs
        for f in [100, 200, 300]:
            y_raw[0, :] += np.exp(-10 * t) * np.sin(2 * np.pi * f * t)

        print("Processing...", end=" ")

        # Process immediately while next measurement is being acquired
        result = processor.process_measurement(r, y_raw)

        print(f"Done! ({len(result['frequencies'])} modes)")

    # Final results
    final_results = processor.get_final_results(
        names=[f"Point_{r}" for r in range(10)],
        output_file="realtime_results.json"
    )

    print(f"\nTotal common modes: {len(final_results['common_f'])}")

    return processor, final_results


def example_with_callbacks():
    """
    Example: Using callbacks to process results as they arrive.
    """
    M_out = 6
    N_use = 28800
    fs = 48000

    processor = StreamingESPRITProcessor(
        M_out=M_out,
        N_use=N_use,
        fs=fs,
        band_index=0,
        L_fraction=0.5,
        K=30
    )

    # Define callback for when each measurement is processed
    def on_measurement_complete(r_index, result):
        """Called when each measurement is processed."""
        print(f"✓ Point {r_index} complete: {len(result['frequencies'])} modes")

        # You could send results to UI, save to database, etc.
        if len(result['frequencies']) > 0:
            dominant_freq = result['frequencies'][np.argmax(result['Q_factors'])]
            print(f"  Dominant mode: {dominant_freq:.2f} Hz")

    # Process measurements
    for r in range(5):
        y_raw = np.random.randn(M_out, N_use) * 0.1
        result = processor.process_measurement(r, y_raw)

        # Trigger callback
        on_measurement_complete(r, result)

    final_results = processor.get_final_results()
    return processor, final_results


if __name__ == "__main__":
    print("Choose example:")
    print("1. Process existing data cube (one-by-one)")
    print("2. Real-time simulation")
    print("3. With callbacks")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        try:
            processor, individual, final = example_with_existing_data()
        except FileNotFoundError:
            print("\nError: esprit_data not found!")
            print("Using real-time simulation instead...\n")
            processor, final = example_real_time_simulation()

    elif choice == "2":
        processor, final = example_real_time_simulation()

    elif choice == "3":
        processor, final = example_with_callbacks()

    else:
        print("Invalid choice, running example 2...")
        processor, final = example_real_time_simulation()
