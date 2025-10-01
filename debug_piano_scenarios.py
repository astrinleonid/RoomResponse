#!/usr/bin/env python3
"""Debug script to test piano_response scenario detection."""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ScenarioManager import ScenarioManager

def main():
    print("=== Piano Response Scenario Debug ===\n")

    # Check current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    print()

    # Check default piano path
    default_piano = os.path.join(cwd, "piano")
    print(f"Default piano path: {default_piano}")
    print(f"  Exists: {os.path.exists(default_piano)}")
    print(f"  Is directory: {os.path.isdir(default_piano)}")
    print()

    # Check relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_piano = os.path.join(script_dir, "piano")
    print(f"Piano path relative to script: {script_piano}")
    print(f"  Exists: {os.path.exists(script_piano)}")
    print(f"  Is directory: {os.path.isdir(script_piano)}")
    print()

    # List contents if exists
    if os.path.isdir(script_piano):
        contents = os.listdir(script_piano)
        print(f"Contents of {script_piano}:")
        for item in contents:
            full_path = os.path.join(script_piano, item)
            item_type = "DIR " if os.path.isdir(full_path) else "FILE"
            print(f"  [{item_type}] {item}")
        print()

    # Test ScenarioManager
    print("Testing ScenarioManager...")
    sm = ScenarioManager()

    # Test with both paths
    for label, test_path in [("CWD-based", default_piano), ("Script-based", script_piano)]:
        print(f"\n--- Testing {label}: {test_path} ---")

        if not os.path.isdir(test_path):
            print("  Path does not exist or is not a directory")
            continue

        # Validate
        is_valid, msg = sm.validate_dataset_root(test_path)
        print(f"  Validation: {is_valid} - {msg}")

        # Analyze
        df = sm.analyze_dataset_filesystem(test_path)
        print(f"  Scenarios found: {len(df)}")

        if not df.empty:
            print("  Scenario details:")
            for idx, row in df.iterrows():
                print(f"    - {row['scenario']}: {row['sample_count']} samples")

if __name__ == "__main__":
    main()
