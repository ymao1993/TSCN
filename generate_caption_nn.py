import argparse
import time
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

from data_manager import DataManager


def load_data_manager(file):
    data_manager = DataManager()
    data_manager.load(file)
    return data_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('metrics', choices=['cosine', 'euclidean'])
    parser.add_argument('num_neighbors', type=int)
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    num_neighbors = args.num_neighbors
    metrics = args.metrics

    dm_train = load_data_manager('data_manager_train_inception.dat')
    dm_val = load_data_manager('data_manager_val_inception.dat')

    if metrics == 'cosine':
        dm_train.normalize_features()

    fi = open(input_file, 'r')
    all_results = []
    for video_segment_name in fi:
        video_segment_name = video_segment_name.strip()
        features = dm_val.get_frames_features(video_segment_name)
        frame_paths = dm_val.get_frames_path(video_segment_name)
        assert(len(features) == len(frame_paths))
        gt_caption = dm_val.get_video_segment_caption(video_segment_name)
        json_frame_results = []

        for i in range(len(features)):
            print('captioning [%s] key frame [%d/%d]...' % (video_segment_name, i+1, len(features)))
            if metrics == 'cosine':
                start_time = time.time()
                captions, knn_frame_paths, k_max_similarities = \
                    dm_train.query_knn_caption_brutal_force_cosine_similarity(features[i], num_neighbors)
                end_time = time.time()
                print('(time cost: %.3f seconds)' % (end_time-start_time))
                json_knn = []
                knn_idx = 0
                for caption, path, similarity in zip(captions, knn_frame_paths, k_max_similarities):
                    json_knn.append({
                        'idx': knn_idx,
                        'path': path,
                        'caption': caption,
                        'similarity': float(similarity)
                    })
                    knn_idx += 1
            else:  # metrics == 'euclidean':
                start_time = time.time()
                captions, knn_frame_paths, k_min_distances = \
                    dm_train.query_knn_caption_brutal_force_euclidean_distance(features[i], num_neighbors)
                end_time = time.time()
                print('(time cost: %.3f seconds)' % (end_time-start_time))
                json_knn = []
                knn_idx = 0
                for caption, path, distance in zip(captions, knn_frame_paths, k_min_distances):
                    json_knn.append({
                        'idx': knn_idx,
                        'path': path,
                        'caption': caption,
                        'distance': float(distance)
                    })
                    knn_idx += 1
            json_frame_result = {
                'path': frame_paths[i],
                'gt_caption': gt_caption,
                'knn': json_knn
            }
            json_frame_results.append(json_frame_result)
        json_video_segment_result = {
            'name': video_segment_name,
            'frame_captions': json_frame_results
        }
        all_results.append([json_video_segment_result])
    fo = open(output_file, 'w')
    json.dump(all_results, fo, indent=4)
    fo.close()
    fi.close()

if __name__ == '__main__':
    main()
