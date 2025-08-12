#!/usr/bin/env python3
"""
Single or Series Room Response Data Collector

Usage (single):
    python collect_dataset.py
    python collect_dataset.py -i
    python collect_dataset.py --quiet --scenario-number 1 --description "Empty room test"

Usage (series):
    python collect_dataset.py --series "0.1,0.2,1-3" --pre-delay 60 --inter-delay 60
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import List

from DatasetCollector import SingleScenarioCollector

# ----------------------------- SDL beep helpers -----------------------------
import math
import numpy as np
import sdl_audio_core as sdl


def _load_sr_from_config(config_file: str, default_sr: int = 48000) -> int:
    try:
        cfg_path = Path(config_file)
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            sr = int(cfg.get("sample_rate", default_sr))
            return sr if sr > 0 else default_sr
    except Exception:
        pass
    return default_sr


def _sdl_beep(config_file: str, count: int = 1, freq: int = 880, dur_ms: int = 200, volume: float = 0.2):
    """
    Play short beeps through SDL using sdl_audio_core.
    Uses sample_rate from recorderConfig.json when available.
    """
    sr = _load_sr_from_config(config_file, 48000)
    n = max(1, int(sr * (dur_ms / 1000.0)))
    t = np.arange(n, dtype=np.float32) / float(sr)
    wave = (volume * np.sin(2.0 * math.pi * float(freq) * t)).astype(np.float32)
    signal = wave.tolist()

    try:
        eng = sdl.AudioEngine()
        cfg = sdl.AudioEngineConfig()
        cfg.sample_rate = sr
        cfg.buffer_size = 1024
        cfg.input_device_id = -1
        cfg.output_device_id = -1
        cfg.enable_logging = False

        if not eng.initialize(cfg):
            raise RuntimeError("AudioEngine.initialize failed")
        if not eng.start():
            raise RuntimeError("AudioEngine.start failed")

        for _ in range(max(1, count)):
            if not eng.start_playback(signal):
                raise RuntimeError("start_playback failed")
            eng.wait_for_playback_completion(dur_ms + 150)
            time.sleep(0.05)  # small gap between beeps

        eng.stop()
        eng.shutdown()
    except Exception:
        # very last resort so flow continues
        for _ in range(max(1, count)):
            print("\a", end="", flush=True)
            time.sleep(dur_ms / 1000.0)
            time.sleep(0.05)


# ----------------------------- CLI parsing -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Room Response Data Collector (single or series)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single
  python collect_dataset.py
  python collect_dataset.py -i
  python collect_dataset.py --quiet --scenario-number 1 --description "Empty room test"

  # Series (comma list and/or ranges; ranges are numeric only)
  python collect_dataset.py --series "0.1,0.2,1-3" --pre-delay 60 --inter-delay 60
        """
    )

    # common
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive audio device selection (default: system defaults)')
    parser.add_argument('--output-dir', type=str, default='room_response_dataset',
                        help='Base output directory (default: room_response_dataset)')
    parser.add_argument('--config-file', type=str, default='recorderConfig.json',
                        help='Recorder configuration file (default: recorderConfig.json)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode - use defaults for names without prompting')

    # single
    parser.add_argument('--scenario-number', type=str, help='Scenario number for single collection')
    parser.add_argument('--description', type=str, help='Scenario description')
    parser.add_argument('--num-measurements', type=int, default=30,
                        help='Measurements per scenario (default: 30)')
    parser.add_argument('--measurement-interval', type=float, default=2.0,
                        help='Seconds between measurements (default: 2.0)')

    # series
    parser.add_argument('--series', type=str,
                        help='Comma-separated scenario numbers and/or numeric ranges (e.g. "0.1,0.2,1-3,7")')
    parser.add_argument('--pre-delay', type=float, default=60.0,
                        help='Delay BEFORE the first scenario (seconds, default: 60)')
    parser.add_argument('--inter-delay', type=float, default=60.0,
                        help='Delay BETWEEN scenarios (seconds, default: 60)')
    parser.add_argument('--series-desc-template', type=str, default='Room response measurement scenario {n}',
                        help='Description template; {n} replaced with scenario number (default shown)')
    parser.add_argument('--beep-volume', type=float, default=0.2, help='Beep volume (0..1, default 0.2)')
    parser.add_argument('--beep-freq', type=int, default=880, help='Beep frequency in Hz (default 880)')
    parser.add_argument('--beep-dur', type=int, default=200, help='Beep duration in ms (default 200)')
    parser.add_argument('--no-beep', action='store_true', help='Disable beeps in series mode')

    return parser.parse_args()


# ----------------------------- Helpers -----------------------------
def _parse_series_expr(expr: str) -> List[str]:
    """
    Parse series string like "0.1,0.2,1-3,7a".
    - Numeric ranges (e.g., 1-3) expand to ['1','2','3']
    - Non-numeric tokens pass through as-is (e.g., '0.1','7a')
    """
    out: List[str] = []
    if not expr:
        return out
    for token in [t.strip() for t in expr.split(',') if t.strip()]:
        if '-' in token:
            a, b = token.split('-', 1)
            if a.replace('.', '', 1).isdigit() and b.replace('.', '', 1).isdigit():
                # numeric range only
                if '.' in a or '.' in b:
                    # float ranges are ambiguous; keep literal token
                    out.append(token)
                else:
                    ai, bi = int(a), int(b)
                    step = 1 if bi >= ai else -1
                    out.extend([str(i) for i in range(ai, bi + step, step)])
            else:
                out.append(token)
        else:
            out.append(token)
    # keep order, remove dups preserving first occurrence
    seen = set()
    uniq = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _load_defaults_from_config(cfg_path: str) -> dict:
    defaults = {"computer": "unknownComp", "room": "unknownRoom"}
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        for k in defaults:
            if k in file_config:
                defaults[k] = file_config[k]
        return defaults
    except Exception:
        return defaults


# ----------------------------- Series collection API -----------------------------
def collect_scenarios_series(
    scenario_numbers: List[str],
    base_output_dir: str,
    config_file: str,
    base_computer: str,
    base_room: str,
    description_template: str = "Room response measurement scenario {n}",
    num_measurements: int = 30,
    measurement_interval: float = 2.0,
    interactive_devices: bool = False,
    pre_delay: float = 60.0,
    inter_delay: float = 60.0,
    enable_beeps: bool = True,
    beep_volume: float = 0.2,
    beep_freq: int = 880,
    beep_dur_ms: int = 200,
):
    """
    Collect a series of scenarios:
      - waits pre_delay BEFORE the first scenario
      - waits inter_delay BETWEEN scenarios
      - plays 1 beep after each scenario, 2 beeps at the end (if enabled)
    """
    print("\n=== SERIES COLLECTION ===")
    print(f"Scenarios: {scenario_numbers}")
    print(f"Base computer: {base_computer} | room: {base_room}")
    print(f"Measurements per scenario: {num_measurements}  |  interval: {measurement_interval}s")
    print(f"Pre-delay: {pre_delay}s  |  Inter-delay: {inter_delay}s  |  Beeps: {'on' if enable_beeps else 'off'}")

    # Pre-delay BEFORE first scenario
    if pre_delay > 0:
        print(f"\nWaiting {pre_delay:.0f}s before starting the first scenario...")
        time.sleep(pre_delay)

    for idx, scen_no in enumerate(scenario_numbers, start=1):
        desc = description_template.replace("{n}", str(scen_no))

        scenario_parameters = {
            "scenario_number": str(scen_no),
            "description": desc,
            "computer_name": base_computer,
            "room_name": base_room,
            "num_measurements": int(num_measurements),
            "measurement_interval": float(measurement_interval),
        }

        print(f"\n--- [{idx}/{len(scenario_numbers)}] Collecting Scenario {scen_no} ---")
        collector = SingleScenarioCollector(
            base_output_dir=base_output_dir,
            recorder_config=config_file,
            scenario_config=scenario_parameters
        )
        # Run collection
        collector.collect_scenario(interactive_devices=interactive_devices)

        # One beep after each scenario
        if enable_beeps:
            _sdl_beep(config_file, count=1, freq=beep_freq, dur_ms=beep_dur_ms, volume=beep_volume)

        # Inter-delay (skip after the last)
        if idx < len(scenario_numbers) and inter_delay > 0:
            print(f"Waiting {inter_delay:.0f}s before next scenario...")
            time.sleep(inter_delay)

    # Double beep at the end
    if enable_beeps:
        _sdl_beep(config_file, count=2, freq=beep_freq, dur_ms=beep_dur_ms, volume=beep_volume)

    print("\n=== SERIES COLLECTION COMPLETE ===")


# ----------------------------- Single-run main -----------------------------
def _run_single(args):
    print("Room Response Single Scenario Data Collector")
    print("=" * 60)

    if args.interactive:
        print("🎛️  Interactive device selection mode enabled")
    else:
        print("🔊 Using default audio devices (use -i for device selection)")

    print(f"\nConfiguration:")
    print(f"  Config file: {args.config_file}")
    print(f"  Output directory: {args.output_dir}")
    if args.quiet:
        print("  Mode: Quiet (using defaults)")

    defaults = _load_defaults_from_config(args.config_file)

    # Gather scenario info
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
            scenario_number = input("Enter scenario number: ").strip() or "1"
        else:
            scenario_number = "1"
    else:
        scenario_number = args.scenario_number
    scenario_parameters["scenario_number"] = scenario_number

    # Description
    if not args.description:
        if not args.quiet:
            description = input("Enter scenario description: ").strip() or f"Room response measurement scenario {scenario_number}"
        else:
            description = f"Room response measurement scenario {scenario_number}"
    else:
        description = args.description
    scenario_parameters["description"] = description

    # Measurements & interval
    scenario_parameters["num_measurements"] = int(args.num_measurements)
    scenario_parameters["measurement_interval"] = float(args.measurement_interval)

    print(f"\nScenario Configuration:")
    print(f"  Computer: {scenario_parameters['computer_name']}")
    print(f"  Room: {scenario_parameters['room_name']}")
    print(f"  Scenario number: {scenario_parameters['scenario_number']}")
    print(f"  Description: {scenario_parameters['description']}")
    print(f"  Measurements: {scenario_parameters['num_measurements']}")
    print(f"  Interval: {scenario_parameters['measurement_interval']}s")

    collector = SingleScenarioCollector(
        base_output_dir=args.output_dir,
        recorder_config=args.config_file,
        scenario_config=scenario_parameters
    )

    collector.collect_scenario(interactive_devices=args.interactive)

    # Offer to store defaults
    try:
        if not args.quiet and input("\nSave room and computer names to config? (y/n): ").strip().lower() == 'y':
            # Update defaults inside config file
            with open(args.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            file_config['computer'] = computer_name
            file_config['room'] = room_name
            with open(args.config_file, 'w', encoding='utf-8') as f:
                json.dump(file_config, f, indent=2)
            print(f"Updated defaults saved to {args.config_file}")
    except Exception as e:
        print(f"Warning: Could not save defaults: {e}")


def main():
    args = parse_arguments()

    # Series mode
    if args.series:
        defaults = _load_defaults_from_config(args.config_file)
        base_computer = defaults['computer']
        base_room = defaults['room']

        # In non-quiet mode, allow overriding names
        if not args.quiet:
            tmp = input(f"\nBase computer name (default {base_computer}): ").strip().replace(" ", "_")
            if tmp:
                base_computer = tmp
            tmp = input(f"Base room name (default {base_room}): ").strip().replace(" ", "_")
            if tmp:
                base_room = tmp

        series_numbers = _parse_series_expr(args.series)
        if not series_numbers:
            print("No valid scenario numbers provided for series.")
            sys.exit(2)

        collect_scenarios_series(
            scenario_numbers=series_numbers,
            base_output_dir=args.output_dir,
            config_file=args.config_file,
            base_computer=base_computer,
            base_room=base_room,
            description_template=args.series_desc_template,
            num_measurements=args.num_measurements,
            measurement_interval=args.measurement_interval,
            interactive_devices=args.interactive,
            pre_delay=args.pre_delay,
            inter_delay=args.inter_delay,
            enable_beeps=not args.no_beep,
            beep_volume=args.beep_volume,
            beep_freq=args.beep_freq,
            beep_dur_ms=args.beep_dur,
        )
        return

    # Single mode
    _run_single(args)


if __name__ == "__main__":
    main()
