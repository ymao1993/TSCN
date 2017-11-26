from data_loader import DataLoader


def write_invalid_video_list(invalid_videos, file_path):
    f = open(file_path, 'w')
    for video_name in invalid_videos:
        f.write(video_name + '\n')
    f.close()


def build_data_loader(vocabulary_file, feature_folder, caption_file, save_path):
    data_loader = DataLoader()
    invalid_videos = data_loader.load_data(vocabulary_file, caption_file, feature_folder)
    data_loader.save(save_path)
    write_invalid_video_list(invalid_videos, save_path + '.invalid.txt')


def main():
    build_data_loader('/home/mscvproject/yu_code/lstm_decoder/data/vocab.txt',
                      '/data01/mscvproject/data/ActivityNetCaptions/train_features_inception',
                      '/home/mscvproject/yu_code/captions/train.json',
                      'data_loader_data/data_loader_train_inception.dat')

    build_data_loader('/home/mscvproject/yu_code/lstm_decoder/data/vocab.txt',
                      '/data01/mscvproject/data/ActivityNetCaptions/val_features_inception',
                      '/home/mscvproject/yu_code/captions/val_1.json',
                      'data_loader_data/data_loader_val1_inception.dat')

if __name__ == '__main__':
    main()
