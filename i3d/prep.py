import sys
import os
import numpy as np
sys.path.insert(0, "../")
import action_recognition.image_funcs as image_funcs


class VideoFramesPreprocessor:
    def __init__(self, mode, resize_size=256, crop_size=224, image_suffix='jpg'):
        self.resize_size = resize_size
        self.crop_size = crop_size
        assert mode in ['train', 'test'], \
            "mode can only either be train or test"
        self.mode = mode
        self.image_suffix = image_suffix

    def construct_frames_blob(self, frames_folder, verbose=False):
        """
        construct a numpy blob of shape (num_frames, crop_size, crop_size, channel_size)
        from the frames folder of video.
        """
        assert os.path.exists(frames_folder), \
            'frame folder %s does not exist.' % frames_folder

        blob = []
        for root, dirs, files in os.walk(frames_folder):
            for file in sorted(files):
                if not file.endswith(self.image_suffix) or file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                if verbose:
                    print('Processing %s...' % file_path)
                image = image_funcs.read_image(file_path, to_float=True)
                image = image_funcs.resize_bilinear_preserve_resoltion(image, self.resize_size)
                crop_shape = (self.crop_size, self.crop_size)
                if self.mode == "train":
                    image = image_funcs.crop_random(image, crop_shape)
                else:
                    image = image_funcs.crop_center(image, crop_shape)
                image = image_funcs.recenter_to_neg_one_and_one(image)
                blob.append(image)
        blob = np.array(blob)
        return blob
