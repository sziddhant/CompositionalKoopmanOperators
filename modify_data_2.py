import os
import h5py
import numpy as np
import shutil
def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt(
        (std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)

def init_stat(dim):
    # Initialize statistics with zeros
    return np.zeros((dim, 3))

def shift_dataset(source_data_dir, dest_data_dir):
    # Initialize sums and counts for states, actions, and attrs
    stat_states = init_stat(4)  # state_dim should be defined globally or passed as a parameter
    stat_actions = init_stat(1)  # action_dim should be defined globally or passed as a parameter
    stat_attrs = init_stat(2)

    sum_states, sum_actions, sum_attrs = None, None, None
    sum_squares_states, sum_squares_actions, sum_squares_attrs = None, None, None
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

            attrs_all, states_all, actions_all = [], [], []

            # Iterate over all h5 files in the source rollout directory
            for filename in sorted(os.listdir(full_source_rollout_dir)):
                if filename.endswith('.h5') and not filename.endswith('.rollout.h5'):
                    source_file_path = os.path.join(full_source_rollout_dir, filename)
                    dest_file_path = os.path.join(full_dest_rollout_dir, filename)

                    # Open the source h5 file and read data
                    with h5py.File(source_file_path, 'r') as source_hf:
                        states = source_hf['states'][:]
                        actions = source_hf['actions'][:]  # Assuming the dataset name is 'actions'
                        attrs = source_hf['attrs'][:]  # Assuming the dataset name is 'attrs'

                        # Subtract the initial position from all positions
                        states[:, 0] -= initial_pos

                        # Save the modified data to the destination file
                        with h5py.File(dest_file_path, 'w') as dest_hf:
                            dest_hf.create_dataset('states', data=states)
                            # If actions and attrs are not modified, you can just copy them.
                            dest_hf.create_dataset('actions', data=actions)
                            dest_hf.create_dataset('attrs', data=attrs)
                        
                        attrs_all.append(attrs)
                        states_all.append(states)
                        actions_all.append(actions)
            
            attrs_all = np.array(attrs_all)
            states_all = np.array(states_all)
            actions_all = np.array(actions_all)

            for datas, stat in zip([attrs_all, states_all, actions_all], [stat_attrs, stat_states, stat_actions]):
                temp_stat = init_stat(stat.shape[0])
                temp_stat[:, 0] = np.mean(datas, axis=(0, 1))[:]
                temp_stat[:, 1] = np.std(datas, axis=(0, 1))[:]
                temp_stat[:, 2] = datas.shape[0]
                stat[:] = combine_stat(stat, temp_stat)

            print(f"Processed and copied rollout directory: {rollout_dir}")

    new_stat_file_path = os.path.join(dest_data_dir, "new_stat.h5")

    # Store the new statistics in a new stat.h5 file
    with h5py.File(new_stat_file_path, 'w') as hf:
        hf.create_dataset('states', data=stat_states)
        hf.create_dataset('actions', data=stat_actions)
        hf.create_dataset('attrs', data=stat_attrs)

    print("New stat.h5 file has been created at", new_stat_file_path)

# Specify the path to your updated data and stat file
source_data_dir = "./data/data_Rope/valid"
dest_data_dir = "./data/data_Rope/valid_fixed"

# Process the dataset and calculate new statistics
shift_dataset(source_data_dir, dest_data_dir)
