import cv2
from utils import constants as cs
from utils import utility, os_utils


if __name__ == '__main__':
    fg_bg = cv2.createBackgroundSubtractorMOG2()
    IMAGE_SIZE = (12, 8)

    path_gen = os_utils.iterate_data(cs.BASE_DATA_PATH + cs.DATA_TRAIN_VIDEOS, ".mp4")

    for path in path_gen:
        utility.write_videos(path, cs.DATA_TRAIN_VIDEOS, cs.DATA_BG_TRAIN_VIDEO)

    path_gen = os_utils.iterate_test_data(cs.BASE_DATA_PATH + cs.DATA_TEST_VIDEOS, ".mp4")
    for path in path_gen:
        utility.write_videos(path, cs.DATA_TEST_VIDEOS, cs.DATA_BG_TEST_VIDEO)
