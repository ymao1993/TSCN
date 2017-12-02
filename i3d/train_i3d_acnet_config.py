num_classes = 200
train_rgb_frames_base_dir = "/nfs01/mscvproject/video_caption/data/ActivityNetClassificationTrim_25fps_256_Frames/train"
train_video_meta_info_file = "preprocess_acnet/acnet_train_meta.txt"
val_rgb_frames_base_dir = "/nfs01/mscvproject/video_caption/data/ActivityNetClassificationTrim_25fps_256_Frames/val"
val_video_meta_info_file = "preprocess_acnet/acnet_val_meta.txt"


pretrained_model_used = 'rgb_imagenet'
pretrained_model_paths = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
}

model_save_name = 'model'
save_model_every_n_iterations = 100
evaluate_model_every_n_iterations = 2

max_iteration = 100000
batch_size = 8
initial_learning_rate = 0.001
momentum = 0.9
