import cv2
import imageio
from utils import os_utils
import utils.constants as cs


def read_image(image_path):
    """
        reads the image

        :param image_path: string
                           image path
    """
    return cv2.imread(image_path)


def convert_gray(image):
    """
        converts bgr image to gray image

        :param image: numpy array
                      gray image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_gaussian(image):
    """
        applies a gaussian filter to the image

        :param image: numpy array

    """
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_canny(image, low_threshold, high_threshold):
    """
            detects edges in the image

            :param image: numpy array

            :param low_threshold: int

            :param high_threshold: int

    """
    return cv2.Canny(image, low_threshold, high_threshold)


def resize(image, shape):
    """
        returns a resize image

        :param image: numpy array
                      image which is to be resize

        :param shape: tuple with exactly two elements (width, height)
                      shape to which image has to be scaled


    """
    return cv2.resize(image, shape)


def equalize_hist(image):
    # =======================
    # Histogram Equalization
    # =======================
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    return image


def show_video_in_window(video_path):
    video = imageio.get_reader(video_path, 'ffmpeg')
    for i in range(len(video)):
        frame = video.get_data(i)
        cv2.line(frame, (400, 0), (400, 1080), (255, 0, 0), 5)
        cv2.line(frame, (1300, 0), (1300, 1080), (255, 0, 0), 5)

        cv2.line(frame, (0, 100), (1920, 100), (255, 0, 0), 5)
        cv2.line(frame, (0, 900), (1920, 900), (255, 0, 0), 5)

        cv2.imshow("frame", frame)

        # =============================
        # Press Q on keyboard to  exit
        # =============================
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def apply_bg_subtraction(video_path):
    fg_bg = cv2.createBackgroundSubtractorMOG2()
    video = imageio.get_reader(video_path, 'ffmpeg')
    for i in range(len(video)):
        frame = video.get_data(i)
        fg_mask = fg_bg.apply(frame)
        cv2.imshow("bg_subtraction", fg_mask)
        # =============================
        # Press Q on keyboard to  exit
        # =============================
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # show_video_in_window("001_001_001.mp4")
    # apply_bg_subtraction("001_001_001.mp4")
    path_gen = os_utils.iterate_data(cs.BASE_DATA_PATH + cs.DATA_TRAIN_VIDEOS, ".mp4")
    for path in path_gen:
        show_video_in_window(path)

