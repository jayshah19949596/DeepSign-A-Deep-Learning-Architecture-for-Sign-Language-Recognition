CONDA_FFMPEG_EXECUTABLE = "/usr/share/applications/anaconda3/bin/ffmpeg"
FFMPEG_EXECUTABLE = "/usr/bin/ffmpeg"

SLASH = "/"

# ==========================
#  All Path constants
# ==========================
BASE_PROJECT_PATH = "/home/jai/Desktop/Thesis2/"
BASE_DATA_PATH = BASE_PROJECT_PATH+"data/"
BASE_LOG_PATH = BASE_PROJECT_PATH+"saved_models/"

# =====================
#  DATA Relative Path
# =====================
DATA_TRAIN_VIDEOS = "videos/training_data"
DATA_TEST_VIDEOS = "videos/testing_data"
DATA_BG_TRAIN_VIDEO = "videos/bg_train_data"
DATA_BG_TEST_VIDEO = "videos/bg_test_data"
DATA_FFMPY_TRAIN_VIDEO = "videos/ffmpy_processed_train_data"
DATA_FFMPY_TEST_VIDEO = "videos/ffmpy_processed_test_data"
DATA_OPEN_POSE_TRAIN = "videos/open_pose_train_data"
DATA_OPEN_POSE_TEST = "videos/open_pose_test_data"

# ===========================
# Saved Models Relative Path
# ===========================
MODEL_VAE = "vae/"
MODEL_SSD = "ssd_mobilenet/"
MODEL_CONV_AE_1 = "auto_encoder_1/"
MODEL_LSTM = "lstm/"
MODEL_LSTM_VAE = "lstm_vae/"
MODEL_BI_LSTM = "bi_lstm/"
MODEL_CONV_LSTM = "conv_lstm/"
MODEL_KERAS_CONV_LSTM = "keras_conv_lstm/"

# ===========================
# PB File Names
# ===========================
ENCODER1_FREEZED_PB_NAME = "encoder1_freezed.pb"
LSTM_FREEZED_PB_NAME = "lstm_freezed.pb"
VAE_FREEZED_PB_NAME = "vae_freezed.pb"
OBJ_DET__PB_NAME = "frozen_inference_graph.pb"

# ===========================
# open pose script file path
# ===========================
OPENPOSE_RUN_SCRIPT_PATH = BASE_PROJECT_PATH+"utils"


# ===========================
# key Constants
# ===========================
FACE = "face_key_points"
HAND = "hand_key_points"
FRAMES = "no_of_frames"
OP_RIGHT_HAND = "hand_right_keypoints"
OP_POSE_POINTS = "pose_keypoints"
OP_PEOPLE = "people"


PEOPLE = "people"
HAND_RIGHT_KPS = "hand_right_keypoints"
POSE_KPS = "pose_keypoints"


# right is right of screen and my left
# ====================
# DTW Results ld and lod
# Accuracy for 1: 0.5821428571428572
# Accuracy for 3: 0.8642857142857143
# Accuracy for 5: 0.9321428571428572
