#!/usr/bin/env python3
"""
Utility to rename scenario numbers in recorded data.

Renames scenario folders and all files inside from one number to another,
handling the folder naming convention: {Computer}-Scenario{Number}-{Room}

Usage:
    python rename_scenario.py <dataset_root> <old_number> <new_number> [--dry-run]

Examples:
    python rename_scenario.py ./piano 45 46
    python rename_scenario.py ./piano 1.5 2 --dry-run
    python rename_scenario.py "C:/data/piano" 7a 7b
"""

import argparse
import re
import sys
from pathlib import Path


# Regex pattern to parse scenario folder names
SCENARIO_FOLDER_RE = re.compile(
    r'^(?P<computer>.+?)-Scenario(?P<num>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)-(?P<room>.+)$',
    re.IGNORECASE,
)


def find_scenario_folder(dataset_root: Path, scenario_number: str) -> Path | None:
    """
    Find a scenario folder by its number within the dataset root.

    Args:
        dataset_root: Path to the dataset directory
        scenario_number: The scenario number to find (e.g., "45", "1.5", "7a")

    Returns:
        Path to the matching folder, or None if not found
    """
    if not dataset_root.is_dir():
        return None

    for item in dataset_root.iterdir():
        if not item.is_dir():
            continue

        match = SCENARIO_FOLDER_RE.match(item.name)
        if match and match.group('num').lower() == scenario_number.lower():
            return item

    return None


def rename_files_in_folder(
    folder: Path,
    old_pattern: str,
    new_pattern: str,
    dry_run: bool = False
) -> tuple[int, int]:
    """
    Recursively rename all files containing old_pattern to use new_pattern.

    Args:
        folder: Path to the folder to process
        old_pattern: The old scenario pattern (e.g., "Ivers&Pond-Scenario88-Take1")
        new_pattern: The new scenario pattern (e.g., "Ivers&Pond-Scenario78-Take1")
        dry_run: If True, only print what would be done

    Returns:
        Tuple of (files_renamed, files_failed)
    """
    files_renamed = 0
    files_failed = 0

    # Collect all files first (to avoid issues with renaming while iterating)
    all_files = list(folder.rglob('*'))

    # Sort by depth (deepest first) to rename files before their parent folders
    all_files.sort(key=lambda p: len(p.parts), reverse=True)

    for file_path in all_files:
        if old_pattern in file_path.name:
            new_name = file_path.name.replace(old_pattern, new_pattern)
            new_path = file_path.parent / new_name

            if dry_run:
                print(f"  Would rename: {file_path.name}")
                print(f"           to: {new_name}")
                files_renamed += 1
            else:
                try:
                    file_path.rename(new_path)
                    print(f"  Renamed: {file_path.name} -> {new_name}")
                    files_renamed += 1
                except OSError as e:
                    print(f"  ERROR renaming {file_path.name}: {e}")
                    files_failed += 1

    return files_renamed, files_failed


def rename_scenario(
    dataset_root: Path,
    old_number: str,
    new_number: str,
    dry_run: bool = False
) -> bool:
    """
    Rename a scenario folder and all its contents from old_number to new_number.

    Args:
        dataset_root: Path to the dataset directory
        old_number: Current scenario number
        new_number: New scenario number
        dry_run: If True, only print what would be done without making changes

    Returns:
        True if successful, False otherwise
    """
    # Find the existing scenario folder
    old_folder = find_scenario_folder(dataset_root, old_number)

    if old_folder is None:
        print(f"Error: No scenario folder found with number '{old_number}' in {dataset_root}")
        return False

    # Parse the folder name to get components
    match = SCENARIO_FOLDER_RE.match(old_folder.name)
    if not match:
        print(f"Error: Could not parse folder name: {old_folder.name}")
        return False

    computer = match.group('computer')
    room = match.group('room')

    # Check if target scenario already exists
    existing_target = find_scenario_folder(dataset_root, new_number)
    if existing_target is not None:
        print(f"Error: Scenario {new_number} already exists: {existing_target.name}")
        return False

    # Construct old and new patterns for file renaming
    old_pattern = f"{computer}-Scenario{old_number}-{room}"
    new_pattern = f"{computer}-Scenario{new_number}-{room}"
    new_folder_name = new_pattern
    new_folder = dataset_root / new_folder_name

    # Print what will be done
    print(f"Renaming scenario {old_number} -> {new_number}")
    print(f"  Folder: {old_folder.name} -> {new_folder_name}")
    print(f"  Path: {dataset_root}")
    print()

    # First, rename all files inside the folder
    print("Renaming files inside folder:")
    files_renamed, files_failed = rename_files_in_folder(
        old_folder, old_pattern, new_pattern, dry_run
    )

    if files_renamed == 0:
        print("  (no files needed renaming)")
    else:
        print(f"\n  Total files renamed: {files_renamed}")
        if files_failed > 0:
            print(f"  Files failed: {files_failed}")

    # Then rename the folder itself
    print(f"\nRenaming folder: {old_folder.name} -> {new_folder_name}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return True

    if files_failed > 0:
        print("\nWarning: Some files failed to rename. Proceeding with folder rename anyway.")

    # Perform the folder rename
    try:
        old_folder.rename(new_folder)
        print(f"\nSuccessfully renamed scenario {old_number} to {new_number}")
        return True
    except OSError as e:
        print(f"\nError renaming folder: {e}")
        return False


def list_scenarios(dataset_root: Path) -> list[tuple[str, str]]:
    """
    List all scenario folders in the dataset root.

    Returns:
        List of (number, folder_name) tuples, sorted by number
    """
    scenarios = []

    if not dataset_root.is_dir():
        return scenarios

    for item in dataset_root.iterdir():
        if not item.is_dir():
            continue

        match = SCENARIO_FOLDER_RE.match(item.name)
        if match:
            scenarios.append((match.group('num'), item.name))

    # Sort by number (try numeric sort, fall back to string sort)
    def sort_key(item):
        num = item[0]
        try:
            # Try to parse as float for numeric sorting
            return (0, float(num), num)
        except ValueError:
            # Fall back to string sorting
            return (1, 0, num)

    return sorted(scenarios, key=sort_key)


def main():
    parser = argparse.ArgumentParser(
        description="Rename scenario numbers in recorded data folders and files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./piano 45 46           Rename scenario 45 to 46
  %(prog)s ./piano 1.5 2 --dry-run Preview renaming 1.5 to 2
  %(prog)s ./piano --list          List all scenarios in the dataset
        """
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "old_number",
        nargs="?",
        help="Current scenario number to rename"
    )
    parser.add_argument(
        "new_number",
        nargs="?",
        help="New scenario number"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all scenarios in the dataset"
    )

    args = parser.parse_args()

    # Resolve the dataset root path
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        print(f"Error: Dataset root does not exist: {dataset_root}")
        sys.exit(1)

    if not dataset_root.is_dir():
        print(f"Error: Dataset root is not a directory: {dataset_root}")
        sys.exit(1)

    # List mode
    if args.list:
        scenarios = list_scenarios(dataset_root)
        if not scenarios:
            print(f"No scenario folders found in {dataset_root}")
        else:
            print(f"Scenarios in {dataset_root}:\n")
            for num, name in scenarios:
                print(f"  {num:>8}  {name}")
            print(f"\nTotal: {len(scenarios)} scenarios")
        sys.exit(0)

    # Rename mode - require both numbers
    if args.old_number is None or args.new_number is None:
        parser.error("Both old_number and new_number are required for renaming")

    # Validate numbers are not empty
    if not args.old_number.strip():
        print("Error: old_number cannot be empty")
        sys.exit(1)

    if not args.new_number.strip():
        print("Error: new_number cannot be empty")
        sys.exit(1)

    # Check if old and new are the same
    if args.old_number.lower() == args.new_number.lower():
        print("Error: old_number and new_number are the same")
        sys.exit(1)

    # Perform the rename
    success = rename_scenario(
        dataset_root,
        args.old_number,
        args.new_number,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
