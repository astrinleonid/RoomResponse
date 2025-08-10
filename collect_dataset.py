#!/usr/bin/env python3
"""
Single Scenario Room Response Data Collector

Usage:
    python collect_scenario.py              # Use default audio devices
    python collect_scenario.py -i           # Interactive device selection
    python collect_scenario.py --interactive # Interactive device selection

This script collects room response data for a single scenario with the naming convention:
<computer_name>-Scenario<number>-<room_name>
"""

import sys
import argparse
import json
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
  python collect_scenario.py --quiet --scenario-number 1 --description "Empty room test"

Dataset naming convention: <computer_name>-Scenario<number>-<room_name>
        """
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Enable interactive audio device selection (default: use system default devices)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='room_response_dataset',
        help='Base output directory (default: room_response_dataset)'
    )

    parser.add_argument(
        '--config-file',
        type=str,
        default='recorderConfig.json',
        help='Recorder configuration file (default: recorderConfig.json)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - use defaults for computer and room names without prompting'
    )

    parser.add_argument(
        '--scenario-number',
        type=str,
        help='Scenario number (if not provided, will prompt user)'
    )

    parser.add_argument(
        '--description',
        type=str,
        help='Scenario description (if not provided, will prompt user)'
    )

    parser.add_argument(
        '--num-measurements',
        type=int,
        default=30,
        help='Number of measurements to collect (default: 30)'
    )

    parser.add_argument(
        '--measurement-interval',
        type=float,
        default=2.0,
        help='Interval between measurements in seconds (default: 2.0)'
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

    print(f"\nConfiguration:")
    print(f"  Config file: {args.config_file}")
    print(f"  Output directory: {args.output_dir}")
    if args.quiet:
        print("  Mode: Quiet (using defaults)")

    # Load defaults from config file
    defaults = {
        "computer": "unknownComp",
        "room": "unknownRoom"
    }

    try:
        with open(args.config_file, 'r') as f:
            file_config = json.load(f)

        for param in defaults.keys():
            if param in file_config:
                defaults[param] = file_config[param]
    except FileNotFoundError:
        print(f"Warning: Config file '{args.config_file}' not found. Using built-in defaults.")
        file_config = {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in config file '{args.config_file}': {e}. Using built-in defaults.")
        file_config = {}

    # Get scenario information
    scenario_parameters = {}

    # Computer name
    computer_name = None
    if not args.quiet:
        computer_name = input(f"\nEnter computer name (default {defaults['computer']}): ").strip().replace(" ", "_")

    if not computer_name:
        computer_name = defaults['computer']
    scenario_parameters["computer_name"] = computer_name

    # Room name
    room_name = None
    if not args.quiet:
        room_name = input(f"Enter room name (default {defaults['room']}): ").strip().replace(" ", "_")

    if not room_name:
        room_name = defaults['room']
    scenario_parameters["room_name"] = room_name

    # Scenario number
    if not args.scenario_number:
        if not args.quiet:
            scenario_number = input("Enter scenario number: ").strip()
            if not scenario_number:
                scenario_number = "1"
        else:
            scenario_number = "1"
    else:
        scenario_number = args.scenario_number
    scenario_parameters["scenario_number"] = scenario_number

    # Description
    if not args.description:
        if not args.quiet:
            description = input("Enter scenario description: ").strip()
            if not description:
                description = f"Room response measurement scenario {scenario_number}"
        else:
            description = f"Room response measurement scenario {scenario_number}"
    else:
        description = args.description
    scenario_parameters["description"] = description

    # Number of measurements
    if not args.quiet:
        try:
            num_measurements = int(input(f"Number of measurements (default {args.num_measurements}): ").strip() or str(args.num_measurements))
        except ValueError:
            num_measurements = args.num_measurements
    else:
        num_measurements = args.num_measurements
    scenario_parameters["num_measurements"] = num_measurements

    # Measurement interval
    if not args.quiet:
        try:
            interval = float(input(f"Interval between measurements in seconds (default {args.measurement_interval}): ").strip() or str(args.measurement_interval))
        except ValueError:
            interval = args.measurement_interval
    else:
        interval = args.measurement_interval
    scenario_parameters["measurement_interval"] = interval

    print(f"\nScenario Configuration:")
    print(f"  Computer: {scenario_parameters['computer_name']}")
    print(f"  Room: {scenario_parameters['room_name']}")
    print(f"  Scenario number: {scenario_parameters['scenario_number']}")
    print(f"  Description: {scenario_parameters['description']}")
    print(f"  Measurements: {scenario_parameters['num_measurements']}")
    print(f"  Interval: {scenario_parameters['measurement_interval']}s")

    try:
        # Create and run collector
        collector = SingleScenarioCollector(
            base_output_dir=args.output_dir,
            recorder_config=args.config_file,
            scenario_config=scenario_parameters
        )

        collector.collect_scenario(interactive_devices=args.interactive)

        # Save updated defaults to config file
        if not args.quiet and input("\nSave room and computer names to config? (y/n): ").strip().lower() == 'y':
            # Update the file_config with new defaults
            file_config['computer'] = computer_name
            file_config['room'] = room_name

            try:
                with open(args.config_file, 'w') as f:
                    json.dump(file_config, f, indent=2)
                print(f"Updated defaults saved to {args.config_file}")
            except Exception as e:
                print(f"Warning: Could not save defaults to config file: {e}")

    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()