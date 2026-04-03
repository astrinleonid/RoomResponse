#!/usr/bin/env python3
"""
Simple script to export point responses with custom output folder
"""
from pathlib import Path
from export_piano_point_responses import export_point_responses

# Configuration
input_folder = Path("piano")
output_folder_name = input("Enter output folder name (default: exported_responses): ").strip() or "exported_responses"
output_folder = Path(output_folder_name)

# Find all Belarus scenarios
scenarios = sorted(input_folder.glob("Belarus-Scenario*"))

if not scenarios:
    print(f"No Belarus-Scenario* folders found in {input_folder}/")
    exit(1)

print(f"\nFound {len(scenarios)} scenarios")
print(f"Output folder: {output_folder.absolute()}")
print("\nStarting export...")

# Export
def progress_print(current, total, message):
    print(f"[{current}/{total}] {message}")

result = export_point_responses(scenarios, output_folder, progress_callback=progress_print)

# Results
print("\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
print(f"Successfully exported: {result['exported_count']} scenarios")
print(f"Failed: {result['failed_count']}")
print(f"Location: {result['output_dir']}")

if result['failed']:
    print("\nFailed exports:")
    for name, error in result['failed']:
        print(f"  - {name}: {error}")
