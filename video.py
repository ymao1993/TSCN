class Frames:
    def __init__(self):
        pass


class Video:
    def __init__(self):
        self.name = ''
        self.frame_features = []
        self.ground_truth_caption = ''

    def batch_iter(self, batch_size):
        pass

    def load_frame_features(self, file_path):
        pass
