#!/usr/bin/env bash
INPUT_FOLDER='/data/caption/data/ActivityNetTrimVideosCaptions_frames/val'
OUTPUT_FOLDER='/data/caption/data/ActivityNetTrimVideosCaptions_features/val'
GPU_IDX=3
BATCH_SIZE=64

python extract_inception.py $INPUT_FOLDER $OUTPUT_FOLDER $BATCH_SIZE $GPU_IDX
