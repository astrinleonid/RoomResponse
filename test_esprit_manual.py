#!/usr/bin/env python3
"""
Manual ESPRIT processing test for existing scenario.
This verifies the import fix works and processes an existing scenario.
"""

from pathlib import Path
from esprit_helper import ESPRITScenarioProcessor

# Process the most recent scenario
scenario_dir = Path("piano/Ivers&Pond-Scenario88-Take1")

print(f"Testing ESPRIT on: {scenario_dir}")
print("=" * 70)

# Default ESPRIT config (matching GUI defaults)
esprit_config = {
    'M_out': 6,
    'N_use': 28800,
    'fs': 48000,
    'L_fraction': 0.5,
    'K': 30,
    'skip_m': 2
}

# Process all 4 bands
result = ESPRITScenarioProcessor.process_scenario(
    scenario_dir=scenario_dir,
    esprit_config=esprit_config,
    force_reprocess=True,
    process_all_bands=True
)

if result:
    print("\n" + "=" * 70)
    print("SUCCESS! ESPRIT processing completed.")
    print("=" * 70)
    print(f"Total modes across all bands: {result.get('total_modes_all_bands', 0)}")
    print("\nBreakdown by band:")
    for band_name, num_modes in result.get('modes_per_band', {}).items():
        print(f"  {band_name}: {num_modes} modes")

    # Check if files were created
    analysis_dir = scenario_dir / "analysis"
    if analysis_dir.exists():
        print(f"\nResults saved to: {analysis_dir}")
        print("Files created:")
        for file in sorted(analysis_dir.glob("esprit_*.json")):
            print(f"  - {file.name}")
else:
    print("\nERROR: ESPRIT processing failed!")
    print("Check the error messages above for details.")
