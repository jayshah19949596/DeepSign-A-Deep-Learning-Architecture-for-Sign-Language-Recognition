import json
import numpy as np
from utils import dtw_helper
import utils.constants as cs
from scipy.spatial.distance import euclidean


def dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = float("inf")
    D0[1:, 0] = float("inf")
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r - 1)
                j_k = min(j + k, c - 1)
                min_list += [D0[i_k, j], D0[i, j_k]]
            D1[i, j] += min(min_list)
    return D1[-1, -1] / sum(D1.shape)


def perform_dtw():
    lab2train = dtw_helper.iterate_data(cs.BASE_PROJECT_PATH + cs.DATA_TRAIN_VIDEOS)
    lab2test = dtw_helper.iterate_data(cs.BASE_PROJECT_PATH + cs.DATA_TEST_VIDEOS)

    test_size = 0
    for test_key in lab2test:
        for test_data in lab2test[test_key]:
            test_size += 1
            distances = []
            for train_key in lab2train:
                total_distance = 0
                for train_data in lab2train[train_key]:
                    total_distance += dtw(train_data, test_data, dist=euclidean)

                    train_od = train_data[0: train_data.shape[0]-1, :] - train_data[1:, :]
                    test_od = test_data[0: test_data.shape[0]-1, :] - test_data[1:, :]

                    total_distance += dtw(train_data, test_data, dist=euclidean)
                    total_distance += dtw(train_od, test_od, dist=euclidean)

                distances.append([total_distance, train_key])

            global correct_1
            correct_1 += evaluate_prediction(test_key, distances, 1)

            global correct_3
            correct_3 += evaluate_prediction(test_key, distances, 3)

            global correct_5
            correct_5 += evaluate_prediction(test_key, distances, 5)

            print(correct_1, correct_3, correct_5, test_size)

    display_accuracy(test_size)


def display_accuracy(test_size):
    """
    Displays the final accuracy
    """
    global correct_1
    print("Accuracy for 1:", (correct_1/test_size))

    global correct_3
    print("Accuracy for 3:", (correct_3/test_size))

    global correct_5
    print("Accuracy for 5:", (correct_5/test_size))


def evaluate_prediction(test_number, distances, top_k):
    """
    counts how many times did DTW gave the correct prediction in top k results

    Parameters
    ----------
    :param test_number: numpy array
                        having a single element which is of type string
                        it indicates the type of sign of the test data

    :param distances: list
                         having a distances of all the training data with a particular test data

    :param top_k: int

    """

    distances = sorted(distances, key=lambda x: x[0])
    distances = distances[0: top_k]
    distances = np.array(distances)
    if test_number in distances[:, 1]:
        return 1
    return 0


if __name__ == '__main__':
    correct_1 = 0
    correct_3 = 0
    correct_5 = 0
    perform_dtw()

