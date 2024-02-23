# Project Setup README

## Introduction
This README provides instructions for preparing datasets and running training sessions for our project.

## Data Preparation

### Generating the Baseline Dataset
Generate the baseline dataset by running the training script train_Rope.sh, with 
--gen_data 1 --obj base

This will create a dataset named `data_baseline_rope`.

### Generating the Object-Centric Dataset
To create the object-centric dataset from the baseline dataset, follow these steps:

1. **Shift the Dataset**:
   Run `modify_data_2.py` to shift the baseline dataset and create `data_obj_rope`. This will also generate a `new_stat.h5` file inside train folder. Move the `new_stat.h5` file from the `train` folder to the main `data_obj_rope` folder, rename it `stat.h5`. Perform these steps for both the `train` and `valid` datasets, but only use `stat.h5` from `train`. Remeber to modify the folder path in modify_data_2.py.

2. **Preprocess the Data**:
    Update the data path in `preprocess_data.py` to point to `data_obj_rope` and run the script: python preprocess_data.py --env Rope


3. **Copy Parameter Files**:
    Run `copy_param.py` to copy the parameter files. Perform this step for both the `train` and `valid` datasets.


## Training Instructions

### Running Baseline Training
To train using the baseline dataset, run `train_Rope.sh` with the `--obj base` option.

### Running Object-Centric Training
For training using the object-centric dataset, run `train_Rope_obj.sh` with the `--obj obj` option.

### Running a Series of Trainings
- For a series of baseline trainings, run `run_baseline.sh`.
- For a series of object-centric trainings, run `run_obj.sh`.

### Generating MPC Videos
Generate MPC videos by running `mpc_Rope.sh`. Remember to modify the model path in `shoot.py` to choose different models for evaluation.

