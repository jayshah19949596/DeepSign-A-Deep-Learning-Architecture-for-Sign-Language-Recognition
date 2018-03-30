import os
import sys
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import utility
import utils.constants as cs
from models.vae import ConVAE
import matplotlib.pyplot as plt
from utils import os_utils, cv_utils
from tensorflow.python.tools import freeze_graph


def iterate_videos(path, input_format):
    """
    Iterates through each file present in path
    and returns a generator that contains the path
    to the video files that has MPEG format

    :param path: string
                 path which has to be iterated

    :param input_format: string
                         data file extension which is to be iterated

    :return full_path: generator
                        contains the path to video files ending with "data_format" extension
    """
    for root, dirs, files in sorted(os.walk(path)):
        files = sorted(files)
        for file in files:
            if file.endswith(input_format):
                full_path = root + cs.Slash + file
                print(full_path)


def get_batch(video_path):
    """

    :param video_path: string
                       path to video from which the batch has to be prepared
    :return: batch_x: numpy array object of shape (batch_size, height, width, d1)
                      array which contains set of frames read from the video
    """
    batch_x = utility.prepare_batch_frames_from_bg_data(video_path, frame_limit=80)
    return batch_x


def display_reconstruction_results(test_frame, reconstructed):
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))

    for images, row in zip([test_frame, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((240, 240)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.1)
    plt.show()


def write_summaries(model_object):
    # Create a summary to monitor training loss tensor
    tf.summary.scalar("loss", tf.reduce_mean(model_object.loss))

    # ==============================================
    # Create a summary to visualize input images
    # ==============================================
    tf.summary.image("input", model_object.inputs_, 40)

    # ===================================================
    # Create a summary to visualize reconstructed images
    # ===================================================
    tf.summary.image("reconstructed", model_object.decoded, 40)

    # ==============================================
    # Create a summary to visualize Loss histogram
    # ==============================================
    tf.summary.histogram("loss_histogram", tf.reduce_mean(model_object.loss))

    return tf.summary.merge_all()


def train():
    loading = False
    logs_path = cs.BASE_LOG_PATH + cs.MODEL_VAE
    tf.reset_default_graph()
    vae = ConVAE()
    vae.build_model()
    epochs = 151
    noise = 0.8
    merged_summary_op = write_summaries(vae)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=10)

    if loading:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logs_path))
        latest_checkpoint_path = tf.train.latest_checkpoint(logs_path)
        checkpoint_number = latest_checkpoint_path.split(".")[0]
        checkpoint_number = int(checkpoint_number.split("_")[-1])
        print("loading checkpoint_number =", checkpoint_number)

    else:
        sess.run(tf.global_variables_initializer())
        checkpoint_number = 0

    summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
    summary_writer.add_graph(sess.graph)

    loop_counter = 1

    for e in tqdm(range(checkpoint_number, checkpoint_number+epochs)):
        path_generator = os_utils.iterate_data(cs.BASE_DATA_PATH+cs.DATA_BG_TRAIN_VIDEO, "mp4")
        batch_counter = 0
        start_time = time.time()

        for video_path in path_generator:
            # ======================================
            # get batches to feed into the network
            # ======================================
            batch_x = get_batch(video_path)

            if batch_x is None:
                # print("video_path", video_path)
                continue

            else:
                print("video_path", video_path)
                print("video number =", batch_counter, "..... batch_x.shape", batch_x.shape,
                      " loop_counter =", checkpoint_number + loop_counter)

                g_loss, l_loss, _, summary = sess.run([vae.generation_loss, vae.latent_loss, vae.opt, merged_summary_op],
                                                      feed_dict={vae.inputs_: batch_x,
                                                                 vae.targets_: batch_x.copy(),
                                                                 vae.noise_var: noise})

            # ==============================
            # Write logs at every iteration
            # ==============================
            summary_writer.add_summary(summary, checkpoint_number + loop_counter)

            print("Epoch: {}/{}...".format(e+1-checkpoint_number, epochs),
                  "Generation loss: {:.4f}".format(np.mean(g_loss)),
                  "Latent loss: {:.4f}".format(np.mean(l_loss)),
                  "Total loss: {:.4f}".format(np.mean(l_loss) + np.mean(g_loss)))

            # if batch_counter % 2 == 0:
            #     print("saving the model at epoch", checkpoint_number + loop_counter)
            #     saver.save(sess, os.path.join(logs_path, 'encoder_epoch_number_{}.ckpt'
            #                                   .format(checkpoint_number + loop_counter)))

            batch_counter += 1
            loop_counter += 1
            if batch_counter == 420:
                end_time = time.time()
                print("==============================================================================================")
                print("Epoch Number", e, "has ended in", end_time-start_time, "seconds for", batch_counter, "videos")
                print("==============================================================================================")

                break

        if e % 10 == 0:
            print("################################################")
            print("saving the model at epoch", checkpoint_number + loop_counter)
            print("################################################")

            saver.save(sess, os.path.join(logs_path, 'encoder_epoch_number_{}.ckpt'
                                          .format(checkpoint_number + loop_counter)))
    # =========================
    # Freeze the session graph
    # =========================
    # freeze_model(sess, logs_path, tf.train.latest_checkpoint(logs_path), cae)
    utility.freeze_model(sess, logs_path, tf.train.latest_checkpoint(logs_path),
                         vae, "vae_train.pb", cs.VAE_FREEZED_PB_NAME)

    print("Run the command line:\n--> tensorboard --logdir={}".format(logs_path),
          "\nThen open http://0.0.0.0:6006/ into your web browser")

    path_generator = os_utils.iterate_test_data(cs.BASE_DATA_PATH+cs.DATA_BG_TRAIN_VIDEO, "mp4")

    for video_path in path_generator:
        test_frame = get_batch(video_path)

        if test_frame is not None:

            test_frame = test_frame[20:40, :, :, :]
            reconstructed = sess.run(vae.decoded, feed_dict={vae.inputs_: test_frame})
            display_reconstruction_results(test_frame, reconstructed)

            break

    sess.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('epochs', type=int,
                        help="Number epochs for which the model is to be trained", default=100)
    parser.add_argument('loading', type=bool,
                        help="Load a pre-trained model or train from scratch", default=False)

    return parser.parse_args(argv)


if __name__ == '__main__':
    total_start_time = time.time()
    train()
    total_end_time = time.time()
    print("===================================================")
    print("Total Execution Time =", total_end_time - total_start_time)
    print("===================================================")

