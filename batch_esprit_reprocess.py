"""
Batch reprocess scenarios with ESPRIT using correct 5-channel configuration.

This script:
1. Identifies all scenarios that need reprocessing (missing 5-channel results)
2. Queues them in numerical order (not alphabetical)
3. Processes each scenario with progress display

Run with: python batch_esprit_reprocess.py
"""
import json
import sys
import re
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from esprit_helper import ESPRITScenarioProcessor


def get_scenario_number(scenario_name: str) -> int:
    """Extract scenario number from name like 'Ivers&Pond-Scenario42-Take1'"""
    match = re.search(r'Scenario(\d+)', scenario_name)
    if match:
        return int(match.group(1))
    return 0


def needs_reprocessing(scenario_dir: Path) -> bool:
    """
    Check if scenario needs reprocessing.
    Returns True if:
    - No ESPRIT results exist, OR
    - Results were processed with wrong channel count (not 5 channels)
    """
    analysis_dir = scenario_dir / 'analysis'
    band0_file = analysis_dir / 'esprit_band0_40-500Hz.json'

    if not band0_file.exists():
        return True

    try:
        with open(band0_file) as f:
            data = json.load(f)

        config = data.get('config', {})
        m_out = config.get('M_out', 0)
        skip_m = config.get('skip_m', 'not_set')

        # Need reprocessing if not 5 channels or skip_m is set
        if m_out != 5:
            return True
        if skip_m is not None and skip_m != 'not_set':
            return True

        return False

    except (json.JSONDecodeError, KeyError):
        return True


def main():
    print("=" * 70)
    print("ESPRIT Batch Reprocessor - 5 Channel Configuration")
    print("=" * 70)
    print()

    # Load config
    config_path = Path('recorderConfig.json')
    if not config_path.exists():
        print("ERROR: recorderConfig.json not found")
        sys.exit(1)

    config = json.load(open(config_path))
    esprit_config = config.get('esprit_config', {})
    esprit_config['fs'] = config['sample_rate']
    esprit_config['N_use'] = int(config['sample_rate'] * config['cycle_duration'])
    esprit_config['M_out'] = config['multichannel_config']['num_channels']

    print("ESPRIT Configuration:")
    print(f"  selected_channels: {esprit_config.get('selected_channels')}")
    print(f"  fs: {esprit_config['fs']} Hz")
    print(f"  K: {esprit_config.get('K', 30)}")
    print()

    # Find all scenarios
    piano_dir = Path('piano')
    if not piano_dir.exists():
        print("ERROR: piano directory not found")
        sys.exit(1)

    all_scenarios = list(piano_dir.glob('Ivers&Pond-Scenario*-Take1'))

    # Sort by scenario NUMBER (not alphabetically)
    all_scenarios.sort(key=lambda p: get_scenario_number(p.name))

    print(f"Found {len(all_scenarios)} total scenarios")
    print()

    # Identify scenarios needing reprocessing
    print("Checking which scenarios need reprocessing...")
    queue = []
    already_done = []

    for scenario_dir in all_scenarios:
        if needs_reprocessing(scenario_dir):
            queue.append(scenario_dir)
        else:
            already_done.append(scenario_dir)

    print(f"  Already processed with 5 channels: {len(already_done)}")
    print(f"  Need reprocessing: {len(queue)}")
    print()

    if not queue:
        print("All scenarios already processed with correct configuration!")
        print("Nothing to do.")
        sys.exit(0)

    # Show queue
    print("Processing queue (numerical order):")
    for i, scenario_dir in enumerate(queue[:10]):
        num = get_scenario_number(scenario_dir.name)
        print(f"  {i+1}. Scenario {num}")
    if len(queue) > 10:
        print(f"  ... and {len(queue) - 10} more")
    print()

    # Ask for confirmation
    response = input(f"Process {len(queue)} scenarios? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        sys.exit(0)

    print()
    print("=" * 70)
    print("Starting batch processing...")
    print("=" * 70)
    print()

    # Process queue
    success_count = 0
    failed_count = 0
    failed_scenarios = []

    for i, scenario_dir in enumerate(queue):
        scenario_num = get_scenario_number(scenario_dir.name)
        print(f"[{i+1}/{len(queue)}] Processing Scenario {scenario_num}...", end=' ', flush=True)

        try:
            result = ESPRITScenarioProcessor.process_scenario(
                scenario_dir=scenario_dir,
                esprit_config=esprit_config,
                force_reprocess=True
            )

            if result:
                # Get total modes from the result
                total_modes = result.get('total_modes_all_bands', 0)
                if total_modes == 0:
                    # Try alternate structure
                    modes_per_band = result.get('modes_per_band', {})
                    total_modes = sum(modes_per_band.values())

                success_count += 1
                print(f"OK ({total_modes} modes)")
            else:
                failed_count += 1
                failed_scenarios.append(scenario_num)
                print("FAILED")

        except Exception as e:
            failed_count += 1
            failed_scenarios.append(scenario_num)
            print(f"ERROR: {e}")

        # Flush output
        sys.stdout.flush()

    # Summary
    print()
    print("=" * 70)
    print("Batch Processing Complete")
    print("=" * 70)
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")

    if failed_scenarios:
        print(f"  Failed scenarios: {failed_scenarios}")

    print()


if __name__ == '__main__':
    main()
