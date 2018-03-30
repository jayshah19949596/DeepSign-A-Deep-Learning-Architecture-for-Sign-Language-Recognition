import tensorflow as tf
from utils import utility
from models.vae import ConVAE
from utils import constants as cs
from tensorflow.python.platform import gfile


def freeze(session, model_path, model, pb_file, freeze_file):
    utility.freeze_model(session, model_path, tf.train.latest_checkpoint(logs_path),
                         model, pb_file, freeze_file)


if __name__ == "__main__":
    logs_path = cs.BASE_LOG_PATH + cs.MODEL_VAE

    vae = ConVAE()
    vae.build_model()

    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(logs_path))
    latest_checkpoint_path = tf.train.latest_checkpoint(logs_path)

    freeze(sess, logs_path, vae, "vae_train.pb", cs.VAE_FREEZED_PB_NAME)

    # frozen_graph_filename = logs_path + cs.VAE_FREEZED_PB_NAME
    #
    # with gfile.FastGFile(frozen_graph_filename, "rb") as f:
    #     graph_def = tf.GraphDef()
    #     byte = f.read()
    #     graph_def.ParseFromString(byte)
    #
    # for node in graph_def.node:
    #     print(node.name)
