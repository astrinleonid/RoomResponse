import os
import shutil
import argparse
import zipfile


def copy_specific_subfolders(source_dir, dest_dir, target_subfolder_name, use_zip=False):
    """
    Copies specific sub-subfolders from a directory structure.
    Can handle either regular folders or zipped folders based on the use_zip parameter.

    Args:
        source_dir (str): The source directory containing the folder structure
        dest_dir (str): The destination directory where the structure will be copied
        target_subfolder_name (str): The name of the sub-subfolder to copy from each subfolder
        use_zip (bool): Whether to process zip files instead of directories
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if use_zip:
        # Process zip files
        zip_files = [f for f in os.listdir(source_dir)
                     if f.endswith('.zip') and os.path.isfile(os.path.join(source_dir, f))]
        items_to_process = zip_files
        print(f"Found {len(zip_files)} zip files to process")
    else:
        # Process directories
        subfolders = [f for f in os.listdir(source_dir)
                      if os.path.isdir(os.path.join(source_dir, f))]
        items_to_process = subfolders
        print(f"Found {len(subfolders)} folders to process")

    # Create a temporary directory for unzipping if needed
    temp_dir = None
    if use_zip:
        temp_dir = os.path.join(source_dir, "_temp_extraction")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    try:
        # Process each item (folder or zip)
        for item in items_to_process:
            item_path = os.path.join(source_dir, item)

            if use_zip:
                # Handle zip file
                subfolder_name = os.path.splitext(item)[0]  # Remove .zip extension

                # Create subfolder in destination
                dest_subfolder_path = os.path.join(dest_dir, subfolder_name)
                if not os.path.exists(dest_subfolder_path):
                    os.makedirs(dest_subfolder_path)

                # Clear temp directory for this zip file
                temp_subfolder_path = os.path.join(temp_dir, subfolder_name)
                if os.path.exists(temp_subfolder_path):
                    shutil.rmtree(temp_subfolder_path)
                os.makedirs(temp_subfolder_path)

                print(f"Extracting {item_path}...")

                # Unzip the file
                with zipfile.ZipFile(item_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_subfolder_path)

                # Check if the target sub-subfolder exists in the unzipped content
                target_path = find_target_subfolder(temp_subfolder_path, target_subfolder_name)

                if target_path:
                    # Copy the target sub-subfolder to the destination
                    dest_target_path = os.path.join(dest_subfolder_path, target_subfolder_name)
                    print(f"Copying {target_path} to {dest_target_path}")

                    copy_directory(target_path, dest_target_path)
                else:
                    print(f"Target subfolder '{target_subfolder_name}' not found in {item}")

            else:
                # Handle regular directory (original behavior)
                subfolder_path = item_path
                subfolder = item

                # Create corresponding subfolder in destination
                dest_subfolder_path = os.path.join(dest_dir, subfolder)
                if not os.path.exists(dest_subfolder_path):
                    os.makedirs(dest_subfolder_path)

                # Check if the target sub-subfolder exists in this subfolder
                target_path = os.path.join(subfolder_path, target_subfolder_name)
                if os.path.exists(target_path) and os.path.isdir(target_path):
                    # Copy the target sub-subfolder to the destination
                    dest_target_path = os.path.join(dest_subfolder_path, target_subfolder_name)
                    print(f"Copying {target_path} to {dest_target_path}")

                    copy_directory(target_path, dest_target_path)
                else:
                    print(f"Target subfolder '{target_subfolder_name}' not found in {subfolder_path}")

    finally:
        # Clean up the temporary directory if it was created
        if use_zip and temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary extraction directory")


def copy_directory(src, dst):
    """
    Helper function to copy a directory with proper Python version handling.
    """
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except TypeError:
        # For Python < 3.8 that doesn't have dirs_exist_ok
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def find_target_subfolder(root_dir, target_name):
    """
    Recursively search for a target subfolder within the directory structure.

    Args:
        root_dir (str): Directory to start searching from
        target_name (str): Name of the subfolder to find

    Returns:
        str or None: Path to the target subfolder if found, None otherwise
    """
    # First check if the target subfolder is at the current level
    potential_target = os.path.join(root_dir, target_name)
    if os.path.exists(potential_target) and os.path.isdir(potential_target):
        return potential_target

    # If not, search in all subdirectories
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            found_path = find_target_subfolder(item_path, target_name)
            if found_path:
                return found_path

    return None

