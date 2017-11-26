from data_manager import DataManager


def build_data_manager(frame_folder, feature_folder, caption_file, key_frame_info_folder, save_path):
    data_manager = DataManager()
    data_manager.load_frame_path_info(frame_folder)
    data_manager.load_captions(caption_file)
    data_manager.load_key_frame_information(key_frame_info_folder)
    data_manager.load_features(feature_folder)
    data_manager.save(save_path)


def main():
    build_data_manager('/data01/mscvproject/data/ActivityNetCaptions/train_frames',
                       '/data01/mscvproject/data/ActivityNetCaptions/train_features_inception',
                       '/home/mscvproject/yu_code/captions/train.json',
                       '/data01/mscvproject/data/ActivityNetCaptions/keyframes/train',
                       'data_manager_train_inception.dat')
    build_data_manager('/data01/mscvproject/data/ActivityNetCaptions/val_frames',
                       '/data01/mscvproject/data/ActivityNetCaptions/val_features_inception',
                       '/home/mscvproject/yu_code/captions/val_1.json',
                       '/data01/mscvproject/data/ActivityNetCaptions/keyframes/val',
                       'data_manager_val_inception.dat')

if __name__ == '__main__':
    main()