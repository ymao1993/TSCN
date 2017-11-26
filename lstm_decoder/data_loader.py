import os
import json
import numpy as np
import cPickle as pickle
import const_config

from vocabulary import Vocabulary

# Max caption length is greater than LSTM truncated length by 1.
# This is because during training, we need input sequence of size
# lstm_truncated_length and target sequence of size lstm_truncated_length.
# And these two sequences are offset by one.
caption_max_length = const_config.lstm_truncated_length + 1


class VideoSegment:
    def __init__(self):
        self.caption = None
        self.caption_mask = np.zeros(caption_max_length)
        self.frame_features = None
        self.frame_paths = []


class Video:
    def __init__(self):
        self.name = ""
        self.video_segments = None
        self.is_valid = False


class DataLoader:
    """
    DataLoader used for training LSTM decoder.
    """
    def __init__(self):
        self.videos = None
        self.vocab = None

    def load_data(self, vocabulary_file, caption_file, feature_folder):
        self.videos = {}
        self._load_vocabulary(vocabulary_file)
        self._load_captions(caption_file)
        self._load_features(feature_folder)
        invalid_videos = self._mark_video_validity()
        self.videos = np.array([val for _, val in self.videos.iteritems()])
        return invalid_videos

    def segmental_sampling(self, batch_size, num_segments):
        """
        Randomly sample a video segment, then perform segmental sampling within the video.
        :return:
        A tuple of numpy arrays.
        (image_features, input_sequence, input_mask, target_sequence)
        image_features is of shape (batch_size, num_segments, feature_length)
        input_sequence is of shape (batch_size, truncated_length)
        input_mask is of shape (batch_size, truncated_length)
        target_sequence is of shape (batch_size, truncated_length)
        """
        num_videos = len(self.videos)
        feature_size = len(self.videos[0].video_segments[0].frame_features[0])
        image_features = np.zeros(shape=(batch_size, num_segments, feature_size))
        input_sequence = np.zeros(shape=(batch_size, const_config.lstm_truncated_length))
        input_mask = np.zeros(shape=(batch_size, const_config.lstm_truncated_length), dtype=np.int)
        target_sequence = np.zeros(shape=(batch_size, const_config.lstm_truncated_length))

        # Sample videos of size batch_size
        # Handle invalid videos by sample rejection.
        valid_video_indices = []
        completed = False
        failure_attempts = 0
        while not completed:
            failure_attempts += 1
            if failure_attempts % 50 == 0:
                print('Warning: %d attempts has been made to sample a mini-batch.' % failure_attempts)
            video_indices = np.random.choice(num_videos, batch_size)
            for video_index in video_indices:
                if self.videos[video_index].is_valid:
                    valid_video_indices.append(video_index)
                    if len(valid_video_indices) == batch_size:
                        completed = True
                        break

        for i, video_idx in enumerate(valid_video_indices):
            video = self.videos[video_idx]
            segment_idx = np.random.randint(low=0, high=len(video.video_segments))
            video_segment = video.video_segments[segment_idx]
            num_frames = len(video_segment.frame_features)
            frame_indices = DataLoader._segmental_sampling(num_frames, num_segments)
            assert video_segment.frame_features is not None,\
                'video_segment.frame_features is None'
            image_features[i, :, :] = [video_segment.frame_features[frame_idx] for frame_idx in frame_indices]
            input_sequence[i, :] = video_segment.caption[:-1]
            input_mask[i, :] = video_segment.caption_mask[:-1]
            target_sequence[i, :] = video_segment.caption[1:]
        return image_features, input_sequence, input_mask, target_sequence

    def segmental_sampling_iter(self, batch_size, num_segments):
        """
        Randomly sample a video segment, then perform segmental sampling within the video.
        Different from segmental_sampling, this method returns an iterator.
        :return:
        A tuple containing:
        (image_features, input_sequence, input_mask, target_sequence, valid_count)
        image_features is of shape (batch_size, num_segments, feature_length)
        input_sequence is of shape (batch_size, truncated_length)
        input_mask is of shape (batch_size, truncated_length)
        target_sequence is of shape (batch_size, truncated_length)
        video_segment_names (batch_size, )
        valid_count
        """
        # Initialize numpy arrays here to avoid repeatedly allocating new buffers.
        feature_size = len(self.videos[0].video_segments[0].frame_features[0])
        image_features = np.zeros(shape=(batch_size, num_segments, feature_size))
        input_sequence = np.zeros(shape=(batch_size, const_config.lstm_truncated_length))
        input_mask = np.zeros(shape=(batch_size, const_config.lstm_truncated_length), dtype=np.int)
        target_sequence = np.zeros(shape=(batch_size, const_config.lstm_truncated_length))
        video_indices = np.zeros(batch_size, dtype=np.int32)
        video_segment_indices = np.zeros(batch_size, dtype=np.int32)
        batch_idx = 0
        num_videos = len(self.videos)
        for i, video in enumerate(self.videos):
            if i % 100 == 0:
                print('Processed [%d/%d] videos.' % (i+1, num_videos))
            if not video.is_valid:
                continue
            for j, video_segment in enumerate(video.video_segments):
                num_frames = len(video_segment.frame_features)
                frame_indices = DataLoader._segmental_sampling(num_frames, num_segments)
                image_features[batch_idx, :, :] =\
                    [video_segment.frame_features[frame_idx] for frame_idx in frame_indices]
                input_sequence[batch_idx, :] = video_segment.caption[:-1]
                input_mask[batch_idx, :] = video_segment.caption_mask[:-1]
                target_sequence[batch_idx, :] = video_segment.caption[1:]
                video_indices[batch_idx] = i
                video_segment_indices[batch_idx] = j
                batch_idx += 1
                if batch_idx == batch_size:
                    yield image_features, input_sequence, input_mask, target_sequence,\
                          video_indices, video_segment_indices, batch_idx
                    batch_idx = 0
        if batch_idx != 0:
            yield image_features, input_sequence, input_mask, target_sequence,\
                  video_indices, video_segment_indices, batch_idx
        return

    def save(self, save_path):
        print('saving data of %d videos...' % len(self.videos))
        file = open(save_path, 'wb')
        data = self.videos
        pickle.dump(data, file)
        file.close()

    def load(self, load_path):
        print('loading data...')
        file = open(load_path, 'rb')
        data = pickle.load(file)
        self.videos = data
        file.close()

    def _load_vocabulary(self, vocabulary_file):
        self.vocab = Vocabulary(vocabulary_file)

    def _load_captions(self, caption_file):
        print('loading captions...')
        file = open(caption_file, 'r')
        database = json.load(open(caption_file, 'r'))
        for video_name in database.keys():
            data = database[video_name]
            sentences = data['sentences']
            video = Video()
            video.name = video_name
            self.videos[video_name] = video
            video.video_segments = []
            for seg_id in range(len(sentences)):
                video_segment = VideoSegment()
                video_segment.caption = self.vocab.sentence_to_id_array(sentences[seg_id])
                caption_length = len(video_segment.caption)
                video_segment.caption_mask[0:min(caption_length, caption_max_length)] = 1
                video_segment.caption.resize(caption_max_length)
                video.video_segments.append(video_segment)
            video.video_segments = np.array(video.video_segments)
        file.close()

    def _load_features(self, feature_folder):
        for root, folder, files in os.walk(feature_folder):
            print('loading features from %s...' % root)
            for file in sorted(files):
                # skip non-txt files
                if not file.endswith('.txt'):
                    continue
                file_path = os.path.join(root, file)
                # extract video name and segment id from file name
                video_segment_name = file[:-4]
                assert len(video_segment_name) >= 14
                video_name = video_segment_name[:13]
                segment_id = int(video_segment_name[13:])
                video = self.videos[video_name]
                video_segment = video.video_segments[segment_id]
                video_segment.frame_features = []
                # load features
                f = open(file_path, 'r')
                for line in f:
                    feature = line.strip().split(',')
                    feature = np.array([float(item) for item in feature])
                    video_segment.frame_features.append(feature)
                video_segment.frame_features = np.array(video_segment.frame_features)
                assert(self.videos[video_name].video_segments[segment_id].frame_features is not None)

    def _mark_video_validity(self):
        total_count = len(self.videos)
        invalid_count = 0
        invalid_videos = []
        for video_name in self.videos:
            video = self.videos[video_name]
            video.is_valid = DataLoader._is_video_valid(video)
            if not video.is_valid:
                invalid_count += 1
                invalid_videos.append(video_name)
        print('%d/%d missing videos found.(%f)' % (invalid_count, total_count, float(invalid_count)/total_count))
        return invalid_videos

    @staticmethod
    def _is_video_valid(video):
        if (video.video_segments is None) or (len(video.video_segments) == 0):
            return False
        if len(video.video_segments) == 0:
            return False
        for video_segment in video.video_segments:
            if (video_segment.frame_features is None) or (len(video_segment.frame_features) == 0):
                return False
        return True

    def _load_frame_path_info(self, frame_folder):
        print('loading frame path info...')
        for root, _, files in os.walk(frame_folder):
            for file in sorted(files):
                # skip non-jpeg files
                if not file.endswith('.jpg'):
                    continue
                file_path = os.path.join(root, file)
                video_segment_name = file_path.split('/')[-2]
                frame_number = int(file.split('.')[0])-1
                video_name = video_segment_name[:13]
                segment_id = int(video_segment_name[13:])
                video = self.videos[video_name]
                video_segment = video.video_segments[segment_id]
                video_segment.frame_paths.append(file_path)
                assert frame_number == len(video_segment.frame_paths)

    @staticmethod
    def _segmental_sampling(total_num, segment_num):
        if segment_num >= total_num:
            results = np.random.choice(total_num, segment_num)
            results.sort()
            return results
        else:
            results = []
            step = int(np.ceil(total_num / segment_num))
            for i in range(segment_num):
                start = step * i
                end = min(start + step, total_num)
                results.append(np.random.randint(low=start, high=end))
            return np.array(results)

