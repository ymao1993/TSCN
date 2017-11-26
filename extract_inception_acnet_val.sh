#!/usr/bin/env bash
INPUT_FOLDER='/data01/mscvproject/data/ActivityNetCaptions/val_frames'
OUTPUT_FOLDER='/data01/mscvproject/data/ActivityNetCaptions/val_features_inception'
GPU_IDX=4
BATCH_SIZE=64

python extract_inception.py $INPUT_FOLDER $OUTPUT_FOLDER $BATCH_SIZE $GPU_IDX
