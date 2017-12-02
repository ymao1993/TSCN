#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="3,4,5,6,7,8,9"
rm -rf train_dir/*
rm -rf summary_dir/*
python train_i3d_acnet.py \
--train_dir train_dir \
--summary_dir summary_dir \
--num_gpus 7 \
--freeze_up_to_logits True
