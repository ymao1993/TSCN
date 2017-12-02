from __future__ import unicode_literals
import os
import subprocess
import argparse


frame_extraction_command = "ffmpeg -i 'VIDEO_PATH' 'FRAME_FOLDER/%06d.jpg'"


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
    parser.add_argument('video_folder')
    parser.add_argument('output_frames_folder')
    parser.add_argument('--video_suffix', default='.mp4')
    parser.add_argument('--image_suffix', default='.jpeg')
    args = parser.parse_args()
    video_folder = args.video_folder
    output_frames_folder = args.output_frames_folder
    video_suffix = args.video_suffix

    actions = os.listdir(video_folder)

    total_count = 0
    failed_count = 0
    for action in actions:
        videos = os.listdir(os.path.join(video_folder, action))
        for video in videos:
            if not video.endswith(video_suffix):
                continue
            total_count += 1
            video_file = os.path.join(video_folder, action, video)
            output_dir = os.path.join(output_frames_folder, video.split('.')[0])
            print('Extracting frames from video %s to %s.' % (video_file, output_dir))
            make_dir(output_dir)
            cmd = frame_extraction_command.replace('VIDEO_PATH', video_file).replace('FRAME_FOLDER', output_dir)
            print cmd
            try:
                subprocess.check_call(cmd, shell=True)
            except subprocess.CalledProcessError:
                failed_count += 1
                print('Failed to execute %s' % cmd)
    print('failed: %d' % failed_count)
    print('total: %d' % total_count)

if __name__ == '__main__':
    main()
