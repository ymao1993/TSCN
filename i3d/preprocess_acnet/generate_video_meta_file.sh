#!/usr/bin/env bash

python generate_video_meta_file.py \
/nfs01/mscvproject/video_caption/data/ActivityNetClassificationTrim_25fps_256_Frames/train \
/nfs01/mscvproject/video_caption/data/ActivityNetClassificationTrim_25fps_256/train \
acnet_train_meta.txt

python generate_video_meta_file.py \
/nfs01/mscvproject/video_caption/data/ActivityNetClassificationTrim_25fps_256_Frames/val \
/nfs01/mscvproject/video_caption/data/ActivityNetClassificationTrim_25fps_256/val \
acnet_val_meta.txt
