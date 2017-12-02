import argparse
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
from data_manager import DataManager
import threading
import Queue
import os

lock = threading.Lock()
total_video_to_be_processed = 0
video_processed = 0


def load_data_manager(file):
    data_manager = DataManager()
    data_manager.load(file)
    return data_manager


def task(q, dm_train, dm_val, metrics, dropout_keep_prob, json_all_results):
    global lock, video_processed, total_video_to_be_processed
    while True:
        video_segment_name = q.get()
        if video_segment_name is None:
            break
        gt_caption = dm_val.get_video_segment_caption(video_segment_name)
        action_class = dm_val.raw_captions_to_class[gt_caption]
        caption = dm_train.query_nearest_caption_by_caption_with_brutal_force(
            gt_caption, metrics=metrics, action_class=action_class, dropout_keep_prob=dropout_keep_prob)
        print('gt_caption:' + gt_caption)
        print('oracle_nn_caption: ' + caption)
        with lock:
            video_processed += 1
            print('[%d/%d]Generated oracle nn caption for %s.' %
                  (video_processed, total_video_to_be_processed, video_segment_name))
            json_all_results.append(
                {
                    'name': video_segment_name,
                    'gt_caption': gt_caption,
                    'video_caption': caption
                }
            )
        q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('metrics', choices=['bleu'])
    parser.add_argument('num_threads', type=int)
    parser.add_argument('--dropout_keep_probs', type=float,
                        help="candidate pool dropout keep probability while doing nearest neighbor search",
                        default=[1.0],
                        nargs='+')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    metrics = args.metrics
    num_threads = args.num_threads
    dropout_keep_probs = args.dropout_keep_probs

    global total_video_to_be_processed, video_processed
    total_video_to_be_processed = sum(1 for _ in open(input_file))
    video_processed = 0
    print('%d videos to be processed' % total_video_to_be_processed)

    # Initialize data loader.
    dm_train = load_data_manager('data_manager_data/data_manager_train_inception.dat')
    dm_val = load_data_manager('data_manager_data/data_manager_val_inception.dat')
    dm_train.tokenize_raw_captions()

    for dropout_keep_prob in dropout_keep_probs:
        # Create task queue.
        q = Queue.Queue(maxsize=100)

        # Start worker threads.
        json_all_results = []
        workers = []
        for i in range(num_threads):
            worker = threading.Thread(target=task,
                                      args=(q, dm_train, dm_val, metrics, dropout_keep_prob, json_all_results))
            worker.start()
            workers.append(worker)

        # Feed data to queue
        fi = open(input_file, 'r')
        for i, video_segment_name in enumerate(fi):
            video_segment_name = video_segment_name.strip()
            q.put(video_segment_name)
        fi.close()

        # Wait for all tasks to be finished.
        q.join()

        # Dump result
        output_file_path = output_file + '_drop_' + str(dropout_keep_prob) + '.json'
        print('Dumping results to %s...' % output_file_path)
        fo = open(output_file_path, 'w')
        json.dump(json_all_results, fo, indent=4)
        fo.close()

        # Stop workers.
        for i in range(num_threads):
            q.put(None)
        for worker in workers:
            worker.join()


if __name__ == '__main__':
    main()
