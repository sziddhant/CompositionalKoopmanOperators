#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --env Rope \
  --pstep 2 \
  --g_dim 32 \
  --len_seq 64 \
  --I_factor 10 \
  --fit_type structured \
	--fit_num 64 \
  --batch_size 64 \
  # --group_size 45\
	# --eval_set demo \
