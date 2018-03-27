FFMPEG_EXECUTABLE = "/usr/share/applications/anaconda3/bin/ffmpeg"
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


# ===========================
# Saved Models Relative Path
# ===========================
MODEL_VAE = "vae/"
MODEL_SSD = "ssd_mobilenet/"
MODEL_CONV_AE_1 = "auto_encoder_1/"
MODEL_LSTM = "lstm/"
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
