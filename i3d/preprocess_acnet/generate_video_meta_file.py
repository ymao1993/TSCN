from __future__ import unicode_literals
import os
import argparse
from random import shuffle


def build_action_map(action_indices_file):
    action_map = {}
    assert os.path.exists(action_indices_file), 'Action indices file does not exist: %s' % action_indices_file
    f = open(action_indices_file)
    for i, action in enumerate(f):
        action_map[action.strip()] = i
    return action_map


def make_dir(directory):
    import os
    import errno
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_folder')
    parser.add_argument('videos_folder')
    parser.add_argument('output_meta_info_file')
    parser.add_argument('--image_suffix', default='.jpg')
    parser.add_argument('--video_suffix', default='.mp4')
    parser.add_argument('--action_indices_file', default='action_indices.txt')
    args = parser.parse_args()
    frames_folder = args.frames_folder
    videos_folder = args.videos_folder
    output_meta_info_file = args.output_meta_info_file
    image_suffix = args.image_suffix
    video_suffix = args.video_suffix
    action_indices_file = args.action_indices_file

    # Build action map.
    action_map = build_action_map(action_indices_file)

    # Build video-action map
    video_to_action = {}
    actions = os.listdir(videos_folder)
    for action in actions:
        print('Building video-action mapping for %s...' % action)
        action_folder = os.path.join(videos_folder, action)
        videos = os.listdir(action_folder)
        for video in videos:
            video_name = video.split(video_suffix)[0]
            video_to_action[video_name] = action

    # Generate Meta file.
    videos = os.listdir(frames_folder)
    lines_to_write = []
    num_videos = len(videos)
    for i, video_name in enumerate(videos):
        print('[%d/%d] Processing %s...' % (i + 1, num_videos, video_name))
        video_frames_folder = os.path.join(frames_folder, video_name)
        files = os.listdir(video_frames_folder)
        image_files = [file for file in files if file.endswith(image_suffix)]
        num_frames = len(image_files)
        # For training I3D model, we only use videos that have more than 64 frames.
        if num_frames >= 64:
            action_name = video_to_action[video_name]
            action_idx = action_map[action_name]
            line = '%s,%d,%d\n' % (video_name, num_frames, action_idx)
            lines_to_write.append(line)
        else:
            print('Skip %s...' % video_name)
    print('Shuffling the meta info...')
    shuffle(lines_to_write)
    print('Dumping the meta info to %s...' % output_meta_info_file)
    fo = open(output_meta_info_file, 'w')
    fo.writelines(lines_to_write)
    fo.close()

if __name__ == '__main__':
    main()
