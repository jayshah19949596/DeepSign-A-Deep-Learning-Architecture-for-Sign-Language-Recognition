import os
from utils import cv_utils
import utils.constants as cs


def create_directory(directory):
    """
    creates a directory

    :param directory: string
                      directory path to be created
    """
    os.makedirs(directory)


def check_existence(directory):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    return os.path.exists(directory)


def iterate_data(directory_path, data_format):
    """
    Iterates through each file present in path
    and returns a generator that contains the path
    to the video files that has particular extension

    :param directory_path: string
                 path which has to be iterated

    :param data_format: string
                        data file extension which is to be iterated

    :return full_path: generator
                        contains the path to video files ending with "data_format" extension
    """
    for root, dirs, files in sorted(os.walk(directory_path)):
        for directory in sorted(dirs):
            for sub_root, sub_dirs, sub_files in (os.walk(os.path.join(root, directory))):
                sub_files = sorted(sub_files)
                for file in sub_files:
                    if file.endswith(data_format) and file != data_format:
                        full_path = sub_root + cs.SLASH + file
                        yield full_path


def iterate_test_data(directory_path, data_format):
    """
    Iterates through each file present in path
    and returns a generator that contains the path
    to the video files that has particular extension

    :param directory_path: string
                 path which has to be iterated

    :param data_format: string
                        data file extension which is to be iterated

    :return full_path: generator
                        contains the path to video files ending with "data_format" extension
    """
    for root, dirs, files in sorted(os.walk(directory_path)):
        for file in files:
            full_path = root + cs.SLASH + file
            yield full_path


if __name__ == '__main__':

    path_gen = iterate_data(cs.BASE_DATA_PATH + cs.DATA_TRAIN_VIDEOS, ".mp4")
    for path in path_gen:
        cv_utils.show_video_in_window(path)
