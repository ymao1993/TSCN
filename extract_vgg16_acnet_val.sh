INPUT_FOLDER='/data01/mscvproject/data/ActivityNetCaptions/val_frames'
OUTPUT_FOLDER='/data01/mscvproject/data/ActivityNetCaptions/val_features'
GPU_IDX=2
BATCH_SIZE=64

python extract_vgg16.py $INPUT_FOLDER $OUTPUT_FOLDER $BATCH_SIZE $GPU_IDX
