import os

def rename_directories(root_dir):
    """
    Renames directories within the specified root directory, replacing spaces with underscores in their names.

    Parameters:
    - root_dir: The root directory within which to rename directories.
    """
    # Walk through the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Iterate over each directory name
        for dirname in dirnames:
            # Check if the directory name contains a space
            if ' ' in dirname:
                # Construct the old and new directory paths
                old_dir_path = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace(' ', '_')
                new_dir_path = os.path.join(dirpath, new_dirname)
                # Rename the directory
                os.rename(old_dir_path, new_dir_path)
                print(f"Renamed '{old_dir_path}' to '{new_dir_path}'")


# Specify the path to the 'results' directory
results_dir = 'results'
# Call the function to start renaming directories
rename_directories(results_dir)
