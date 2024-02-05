#!/usr/bin/env bash

# Directory for logs
# mkdir -p logs_baseline

# Loop over the seeds
for seed in {1..25}
do
   CUDA_VISIBLE_DEVICES=0 python train_object.py \
    --env Rope \
    --len_seq 64 \
    --I_factor 10 \
    --batch_size 32 \
    --fit_num 32 \
    --lr 1e-4 \
    --g_dim 32 \
    --pstep 2 \
    --fit_type structured \
    --gen_data 0 \
    --group_size 5 \
    --num_workers 1 \
    --obj obj \
    --seed "$seed" \
    # --data_dir data/data_baseline_Rope > "logs_baseline/train_baseline_seed_${seed}.log" 2>&1 &
done

# Wait for all processes to finish
wait
