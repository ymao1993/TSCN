from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cv2 import DualTVL1OpticalFlow_create as DualTVL1
from tensorflow.python.platform import flags
import os
import sys

sys.path.insert(0, '..')
from utils import image_funcs

import numpy as np


def make_dir(directory):
    import os
    import errno
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


FLAGS = flags.FLAGS
flags.DEFINE_string('expected_rgb_frame_suffix', ".jpg", 'Expected RGB frame files\' suffix.')


def compute_optical_flow_tvl1(video_frames_folder):
    """Compute the TV-L1 optical flow."""
    TVL1 = DualTVL1()

    # Collect RGB frame paths.
    rgb_frame_files = os.listdir(video_frames_folder)
    rgb_frame_files = [frame_file for frame_file in rgb_frame_files
                       if frame_file.endswith(FLAGS.expected_rgb_frame_suffix)]
    rgb_frame_files.sort()
    num_frames = len(rgb_frame_files)
    assert num_frames >= 2, "Only find %d (<=2) RGB frames under %s." % (num_frames, video_frames_folder)

    # Iteratively compute optical flows.
    optical_flows = []
    prev_frame = image_funcs.rgb_to_gray(image_funcs.read_image(rgb_frame_files[0], to_float=False))
    for i in range(1, num_frames):
        cur_frame = image_funcs.rgb_to_gray(image_funcs.read_image(rgb_frame_files[1], to_float=False))
        cur_flow = TVL1.calc(prev_frame, cur_frame, None)
        assert (cur_flow.dtype == np.float32)
        optical_flows.append(cur_flow)
        prev_frame = cur_frame
    return optical_flows


def save_images(images, file_pattern, start_idx=1):
    for i, image in enumerate(images):
        file_path = file_pattern % (start_idx+i)
        make_dir(file_path)
        image_funcs.save_image(image, file_path)


def main():
    optical_flows = compute_optical_flow_tvl1("")
    frame_file_pattern = "%05d.jpg"
    folder = ""
    file_pattern = os.path.join(folder, frame_file_pattern)
    save_images(optical_flows, file_pattern)


if __name__ == '__main__':
    main()
