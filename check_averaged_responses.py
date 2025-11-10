#!/usr/bin/env python3
"""
Check which piano scenarios have averaged responses
"""

import os
from pathlib import Path

piano_dir = Path("piano")

print("Checking averaged responses in piano scenarios:")
print("=" * 70)

scenarios = sorted(piano_dir.glob("Neumann-Scenario*"))

has_avg = []
missing_avg = []

for scenario in scenarios:
    scenario_num = scenario.name.split("-Scenario")[1].split("-")[0]
    avg_dir = scenario / "averaged_responses"

    if avg_dir.exists():
        avg_files = list(avg_dir.glob("*.npy"))
        has_avg.append((scenario_num, scenario.name, len(avg_files)))
        print(f"[YES] Scenario {scenario_num:>3}: {len(avg_files)} averaged files")
    else:
        missing_avg.append((scenario_num, scenario.name))
        print(f"[NO]  Scenario {scenario_num:>3}: NO averaged_responses folder")

print("=" * 70)
print(f"\nSummary:")
print(f"  Scenarios with averaged responses: {len(has_avg)}")
print(f"  Scenarios missing averaged responses: {len(missing_avg)}")

if missing_avg:
    print(f"\nMissing averaged responses for scenarios:")
    for num, name in missing_avg:
        print(f"  - Scenario {num} ({name})")
