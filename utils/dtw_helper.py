import os
import json
import numpy as np
import utils.constants as cs
from collections import defaultdict


def load_json_file(file_full_path):
    with open(file_full_path) as file:
        json_data = json.load(file)
    return json_data


def pre_process(root2file):
    lab2data = defaultdict(list)
    for root in root2file:
        video_data = []
        video_num = int(root.split(cs.SLASH)[-2][0:3])
        # print(video_num)
        for file in root2file[root]:
            # print(file)
            json_data = load_json_file(root+file)
            face_x = json_data[cs.PEOPLE][0][cs.POSE_KPS][3 * 0]
            face_y = json_data[cs.PEOPLE][0][cs.POSE_KPS][(3 * 0) + 1]
            x = json_data[cs.PEOPLE][0][cs.POSE_KPS][3 * 4]
            y = json_data[cs.PEOPLE][0][cs.POSE_KPS][(3 * 4)+1]
            video_data.append([x-face_x, y-face_y])
        lab2data[video_num].append(np.array(video_data))
    return lab2data


def iterate_data(directory_path):

    root2file = {}
    for root, dirs, files in sorted(os.walk(directory_path)):
        if files:
            root2file[root+cs.SLASH] = sorted(files)
    return pre_process(root2file)
    # print(root2file)


if __name__ == '__main__':
    lab2train = iterate_data(cs.BASE_DATA_PATH + cs.DATA_OPEN_POSE_TRAIN)
    lab2test = iterate_data(cs.BASE_DATA_PATH + cs.DATA_OPEN_POSE_TEST)
