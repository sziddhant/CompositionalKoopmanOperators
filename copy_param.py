import os
import shutil

def copy_param_files(source_data_dir, dest_data_dir):
    # Check if destination directory exists, create it if not
    if not os.path.exists(dest_data_dir):
        os.makedirs(dest_data_dir)

    # Iterate over all files in the source directory
    for root, dirs, files in os.walk(source_data_dir):
        for file in files:
            # Check if the file extension is .param
            if file.endswith('.param'):
                # Construct the full path to the file
                source_file_path = os.path.join(root, file)
                
                # Construct the destination path. This keeps the directory structure.
                destination_path = os.path.join(dest_data_dir, os.path.relpath(root, source_data_dir), file)
                
                # Create intermediate directories if they don't exist
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                
                # Copy the file to the destination directory
                shutil.copy2(source_file_path, destination_path)

# Specify the path to your original and new data directories
source_data_dir = "./data/data_Rope/valid"
dest_data_dir = "./data/data_Rope/valid_fixed"

# Copy all .param files
copy_param_files(source_data_dir, dest_data_dir)
