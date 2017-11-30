from __future__ import unicode_literals
import os
import argparse


def build_action_map(action_indices_file):
    action_map = {}
    assert os.path.exists(action_indices_file), 'Action indices file does not exist: %s' % action_indices_file
    f = open(action_indices_file)
    for i, action in enumerate(f):
        action_map[action] = i
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
    parser.add_argument('output_meta_info_file')
    parser.add_argument('--image_suffix', default='.jpeg')
    parser.add_argument('--action_indices_file', default='action_indices.txt')
    args = parser.parse_args()
    frames_folder = args.frames_folder
    output_meta_info_file = args.output_meta_info_file
    image_suffix = args.image_suffix
    action_indices_file = args.action_indices_file

    # Build action map.
    action_map = build_action_map(action_indices_file)
    actions = os.listdir(frames_folder)

    fo = open(output_meta_info_file, 'w')
    for action in actions:
        print('Processing %s...' % action)
        videos = os.listdir(os.path.join(frames_folder, action))
        for video in videos:
            video_frames_folder = os.path.join(frames_folder, action, video)
            image_files = []
            for root, _, files in os.walk(video_frames_folder):
                for file in files:
                    if file.endswith(image_suffix):
                        image_files.append(file)
            num_frames = len(image_files)
            line = '%s,%d,%d\n' % (video, num_frames, action_map[action])
            fo.write(line)
    fo.close()

if __name__ == '__main__':
    main()
