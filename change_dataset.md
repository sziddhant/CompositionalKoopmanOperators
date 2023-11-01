change between baseline and object_centric version:

run modify_data_2.py with correct data folder (to generate i/t.h5 and new_stat.h5)
run copy_param.py to make sure using the same set of params
run preprocess_data.py with --env Rope --obj _object_centric or anything not _baseline, so that it generates i.rollout.h5

OR
just use data_object_centric in train to generate entire object_centric dataset, make sure to use the same random seed