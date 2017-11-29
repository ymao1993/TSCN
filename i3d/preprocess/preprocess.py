from __future__ import unicode_literals
import os
import subprocess
import argparse


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
    parser.add_argument('video_base_dir')
    parser.add_argument('output_video_base_dir')
    parser.add_argument('--expected_num_actions', default=200)
    parser.add_argument('--expected_video_suffix', default='.mp4')
    args = parser.parse_args()
    base = args.video_base_dir
    output_base = args.output_video_base_dir
    expected_num_actions = args.expected_num_actions
    expected_video_suffix = args.expected_video_suffix

    # Get actions.
    actions = os.listdir(base)

    assert len(actions) == expected_num_actions, \
        'Number of folders found under %s does not equal to the expected number of actions %d' %\
        (base, expected_num_actions)

    # Command that does the following thing:
    # 1. Sample videos at 25 fps.
    # 2. Resize the image bi-linearly and keep the original aspect of ratio so that the smallest dimension is 256.
    resample_command = """ffmpeg -y -i 'FILE' -r 25 -vf scale="'if(gt(iw,ih),-2,256)':'if(gt(ih,iw),-2,256)'" -an '@&OUT'"""

    for action in actions:
        videos = os.listdir(os.path.join(base, action))
        for video in videos:
            if not video.endswith(expected_video_suffix):
                continue
            # Re-sample to 25FPS and scale smallest side to 256
            video_file_path = os.path.join(base, action, video)
            print('Processing %s' % video_file_path)
            output_video_file_path = os.path.join(output_base, action, video)
            make_dir(os.path.join(output_base, action))
            if not os.path.exists(output_video_file_path):
                cmd = resample_command.replace('FILE', video_file_path).replace('@&OUT', output_video_file_path)
                print cmd
                subprocess.check_call(cmd, shell=True)

if __name__ == '__main__':
    main()
