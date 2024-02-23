#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_object.py \
	--env Rope \
	--len_seq 64 \
	--I_factor 10 \
	--batch_size 16 \
	--fit_num 16 \
	--lr 1e-4 \
	--g_dim 32 \
	--pstep 2 \
	--fit_type structured \
	--gen_data 0\
    --group_size 5\
	--num_workers 1\
    --obj obj\