import numpy as np
import prep


def main():
    frame_processor = prep.VideoFramesPreprocessor('test')
    blob = frame_processor.construct_frames_blob('example/v_xCplsH6deic0')
    blobs = np.array([blob])
    print('dumping result...')
    blobs.dump('example/v_xCplsH6deic0.npy')


if __name__ == '__main__':
    main()

