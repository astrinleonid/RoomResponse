#!/usr/bin/env python3
"""
Single Scenario Room Response Data Collector

Usage:
    python collect_scenario.py              # Use default audio devices
    python collect_scenario.py -i           # Interactive device selection
    python collect_scenario.py --interactive # Interactive device selection

This script collects room response data for a single scenario with the naming convention:
<computer_name>_Scenario<number>_<room_name>
"""

import sys
import argparse
from DatasetCollector import SingleScenarioCollector


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Room Response Single Scenario Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_scenario.py                    # Use default audio devices
  python collect_scenario.py -i                 # Interactive device selection
  python collect_scenario.py --interactive      # Interactive device selection

Dataset naming convention: <computer_name>_Scenario<number>_<room_name>
        """
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Enable interactive audio device selection (default: use system default devices)'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        help='Audio sample rate in Hz (default: 48000)'
    )

    parser.add_argument(
        '--pulse-duration',
        type=float,
        default=0.008,
        help='Pulse duration in seconds (default: 0.008)'
    )

    parser.add_argument(
        '--cycle-duration',
        type=float,
        default=0.1,
        help='Cycle duration in seconds (default: 0.1)'
    )

    parser.add_argument(
        '--num-pulses',
        type=int,
        default=8,
        help='Number of pulses in test signal (default: 8)'
    )

    parser.add_argument(
        '--volume',
        type=float,
        default=0.6,
        help='Playback volume 0.0-1.0 (default: 0.4)'
    )

    parser.add_argument(
        '--impulse-form',
        choices=['square', 'sine'],
        default='sine',
        help='Pulse shape (default: sine)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='room_response_dataset',
        help='Base output directory (default: room_response_dataset)'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    print("Room Response Single Scenario Data Collector")
    print("=" * 60)

    if args.interactive:
        print("🎛️  Interactive device selection mode enabled")
    else:
        print("🔊 Using default audio devices")
        print("   (Use -i or --interactive for device selection)")

    # Create recorder configuration from arguments
    recorder_config = {
        'sample_rate': args.sample_rate,
        'pulse_duration': args.pulse_duration,
        'pulse_fade': 0.0001,  # Keep default fade
        'cycle_duration': args.cycle_duration,
        'num_pulses': args.num_pulses,
        'volume': args.volume,
        'impulse_form': args.impulse_form
    }

    scenario_parameters = {}
    # Get scenario information
    computer_name = input("Enter computer name: ").strip().replace(" ", "_")
    if not computer_name:
        computer_name = "LeonidDesctop"
    scenario_parameters["computer_name"] = computer_name

    room_name = input("Enter room name: ").strip().replace(" ", "_")
    if not room_name:
        room_name = "SmallSPb"
    scenario_parameters["room_name"] = room_name

    try:
        scenario_number = (input("Enter scenario number: ").strip())
    except ValueError:
        print("Invalid scenario number, using 1")
        scenario_number = "0.1"
    scenario_parameters["scenario_number"] = scenario_number

    description = input("Enter scenario description: ").strip()
    if not description:
        description = f"Room response measurement scenario {scenario_number}"
    scenario_parameters["description"] = description

    try:
        num_measurements = int(input(f"Number of measurements (default 100): ").strip() or "100")
    except ValueError:
        num_measurements = 100
    scenario_parameters["num_measurements"] = num_measurements

    try:
        interval = float(input(f"Interval between measurements in seconds (default 1.0): ").strip() or "1.0")
    except ValueError:
        interval = 1.0
    scenario_parameters["measurement_interval"] = interval



    # Print configuration
    print(f"\nRecorder Configuration:")
    print(f"  Sample rate: {recorder_config['sample_rate']} Hz")
    print(f"  Pulse duration: {recorder_config['pulse_duration'] * 1000:.1f} ms")
    print(f"  Cycle duration: {recorder_config['cycle_duration'] * 1000:.1f} ms")
    print(f"  Number of pulses: {recorder_config['num_pulses']}")
    print(f"  Volume: {recorder_config['volume']}")
    print(f"  Pulse shape: {recorder_config['impulse_form']}")
    print(f"  Output directory: {args.output_dir}")

    try:
        # Create and run collector
        collector = SingleScenarioCollector(
            base_output_dir=args.output_dir,
            recorder_config=recorder_config,
            scenario_config=scenario_parameters
        )

        collector.collect_scenario(interactive_devices=args.interactive)

    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()