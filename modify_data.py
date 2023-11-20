import os
import h5py
import numpy as np
import shutil

def shift_dataset(source_data_dir, dest_data_dir):
    sum_states = None
    sum_squares_states = None
    count = 0
    # Check if destination directory exists, create it if not
    if not os.path.exists(dest_data_dir):
        os.makedirs(dest_data_dir)

    # Iterate over all rollout directories in the dataset
    for rollout_dir in os.listdir(source_data_dir):
        full_source_rollout_dir = os.path.join(source_data_dir, rollout_dir)
        full_dest_rollout_dir = os.path.join(dest_data_dir, rollout_dir)

        if os.path.isdir(full_source_rollout_dir):
            initial_pos = None

            # Copy the directory structure and non-h5 files
            if not os.path.exists(full_dest_rollout_dir):
                shutil.copytree(full_source_rollout_dir, full_dest_rollout_dir, ignore=shutil.ignore_patterns('*.h5'))

            initial_file_path = os.path.join(full_source_rollout_dir, '0.h5')
            with h5py.File(initial_file_path, 'r') as initial_hf:
                initial_states = initial_hf['states'][:]  # Assuming the dataset name is 'states'
                initial_pos = initial_states[0, 0].copy()  # Get the initial position of the first ball

            # Iterate over all h5 files in the source rollout directory
            for filename in sorted(os.listdir(full_source_rollout_dir)):
                if filename.endswith('.h5') and not filename.endswith('.rollout.h5'):
                    source_file_path = os.path.join(full_source_rollout_dir, filename)
                    dest_file_path = os.path.join(full_dest_rollout_dir, filename)

                    # Open the source h5 file and read data
                    with h5py.File(source_file_path, 'r') as source_hf:
                        states = source_hf['states'][:]  # Assuming the dataset name is 'states'

                        # Subtract the initial position from all positions
                        states[:, 0] -= initial_pos

                        # Save the modified data to the destination file
                        with h5py.File(dest_file_path, 'w') as dest_hf:
                            dest_hf.create_dataset('states', data=states)
                        
                        # Accumulate data for new stat.h5
                        if sum_states is None:
                            sum_states = np.sum(states, axis=0)
                            sum_squares_states = np.sum(np.square(states), axis=0)
                        else:
                            sum_states += np.sum(states, axis=0)
                            sum_squares_states += np.sum(np.square(states), axis=0)
                        count += states.shape[0]  # Update count

            print(f"Processed and copied rollout directory: {rollout_dir}")
            if int(rollout_dir) > 100:
                break

    # Calculate new statistics
    mean_states = sum_states / count
    var_states = (sum_squares_states / count) - np.square(mean_states)
    std_states = np.sqrt(var_states)
    
    return mean_states, var_states, std_states, count

# Specify the path to your updated data and stat file
source_data_dir = "./data/data_Rope/train"
dest_data_dir = "./data/data_Rope/train_fixed"
# stat_file_path = "path/to/your/data_rope/stat.h5"
state_dim = 4  # Replace with your actual state dimension

# Calculate new statistics for states
mean_states, var_states, std_states, count = shift_dataset(source_data_dir, dest_data_dir)
new_stats = np.array([mean_states, std_states, np.array([count] * len(mean_states))]).T
# Update 'states' statistics in stat.h5 file
# new_state_stat = np.stack([mean_states, std_states, np.full(state_dim, count)], axis=-1)
print(new_stats)