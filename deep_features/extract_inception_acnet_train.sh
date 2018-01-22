#!/usr/bin/env bash
INPUT_FOLDER='/data/caption/data/ActivityNetTrimVideosCaptions_frames/train'
OUTPUT_FOLDER='/data/caption/data/ActivityNetTrimVideosCaptions_features/train'
GPU_IDX=0
BATCH_SIZE=64

python extract_inception.py $INPUT_FOLDER $OUTPUT_FOLDER $BATCH_SIZE $GPU_IDX
