num_classes = 200
rgb_frames_base_dir = ""
video_meta_info_file = ""
pretrained_model_paths = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
}

model_save_name = 'model'
save_model_every_n_iterations = 1000
save_summary_every_n_iterations = 10

max_iteration = 1000
batch_size = 8
initial_learning_rate = 0.01
momentum = 0.9

decay_every_n_steps = 1000

