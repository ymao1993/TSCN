INPUT_FOLDER='/data01/mscvproject/data/ActivityNetCaptions/train_frames'
OUTPUT_FOLDER='/data01/mscvproject/data/ActivityNetCaptions/train_features'
GPU_IDX=1
BATCH_SIZE=64

python extract_vgg16.py $INPUT_FOLDER $OUTPUT_FOLDER $BATCH_SIZE $GPU_IDX
