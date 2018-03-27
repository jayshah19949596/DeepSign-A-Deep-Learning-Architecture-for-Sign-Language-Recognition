import os
import time
import random
import numpy as np
import tensorflow as tf
import utils.constants as cs
from sklearn import preprocessing
from models.lstm import RecurrentNetwork
from tensorflow.python.platform import gfile
from utils import utility, os_utils, cv_utils
from sklearn.preprocessing import OneHotEncoder


def get_batch(video_path):
    # batch_x = cv_utils.prepare_batch_frames(video_path, all_frame=all_frame)
    batch_x = utility.prepare_batch_frames_from_bg_data(video_path=video_path, frame_limit=50)
    return batch_x


def get_target_name(video_path):
    # print(video_path)
    split_path = video_path.split(cs.SLASH)
    # print(int(split_path[-1][0:3]))
    return int(split_path[-1][0:3])


def get_label_enocder(path_gen):
    list_of_target_names = []
    one_hot_list = []

    counter = 1
    for video_path in path_gen:
        # batch_x = get_batch(video_path, False)
        #
        # if batch_x is None:
        #     continue

        list_of_target_names.append(get_target_name(video_path))
        one_hot_list.append(get_target_name(video_path))
        counter += 1

        # if counter == 10:
        #     break
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(one_hot_list)
    transformed = label_encoder.transform(one_hot_list)
    return label_encoder, len(transformed)


def get_encoded_embeddings(logs_path):

    frozen_graph_filename = logs_path + cs.ENCODER1_FREEZED_PB_NAME

    with gfile.FastGFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        byte = f.read()
        graph_def.ParseFromString(byte)

    # for node in graph_def.node:
    #     print(node.name)
    tf.import_graph_def(graph_def, name='')

    detection_graph = tf.get_default_graph()
    x = detection_graph.get_tensor_by_name('inputs:0')
    encoded = detection_graph.get_tensor_by_name('encoder/encoded/LeakyRelu/Maximum:0')

    # embedding = sess.run(encoded, feed_dict={x: frame})
    # embedding = embedding.reshape((1, / len(sampling_list)embedding.shape[0], embedding.shape[1]))

    return x, encoded


def write_summaries(validation_acc, loss):
    # ================================================
    # Create a summary to monitor training loss tensor
    # ================================================
    tf.summary.scalar("loss", loss)

    # ================================================
    # Create a summary to monitor validation accuracy
    # ================================================
    tf.summary.scalar("validation_accuracy", validation_acc)

    return tf.summary.merge_all()


def train():
    epochs = 50
    sampling_number = 70
    encoder_logs_path = cs.BASE_LOG_PATH + cs.MODEL_CONV_AE_1
    path_generator = os_utils.iterate_data(cs.BASE_DATA_PATH + cs.DATA_BG_TRAIN_VIDEO, "mp4")
    logs_path = cs.BASE_LOG_PATH + cs.MODEL_LSTM
    checkpoint_number = 0
    loop_counter = 1

    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():
        val_acc = tf.Variable(0.0, tf.float32)
        tot_loss = tf.Variable(0.0, tf.float32)

        rnn = RecurrentNetwork(lstm_size=128, batch_len=BATCH_SIZE, output_nodes=14, learning_rate=0.001)
        rnn.build_model()
        stage_1_ip, stage_2_ip = get_encoded_embeddings(encoder_logs_path)
        prediction = tf.argmax(rnn.predictions, 1)

    label_encoder, num_classes = get_label_enocder(path_generator)

    with graph.as_default():
        merged_summary_op = write_summaries(val_acc, tot_loss)
        summary_writer = tf.summary.FileWriter(logs_path, graph=graph)
        summary_writer.add_graph(graph)
        saver = tf.train.Saver(max_to_keep=4)

    loop_counter = 1

    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())
        iteration = 1
        tf.get_default_graph().finalize()
        for e in range(epochs):
            sampling_list = random.sample(range(0, 419), sampling_number)
            start_time = time.time()
            total_loss = 0
            validation_accuracy = 0
            state = sess.run(rnn.initial_state)

            path_generator = os_utils.iterate_data(cs.BASE_DATA_PATH + cs.DATA_BG_TRAIN_VIDEO, "mp4")

            batch_counter = 0
            for video_path in path_generator:

                batch_x = get_batch(video_path)
                batch_y = get_target_name(video_path)

                if batch_x is None:
                    continue

                encoded_batch = sess.run(stage_2_ip, feed_dict={stage_1_ip: batch_x})
                encoded_batch = encoded_batch.reshape((1, encoded_batch.shape[0], encoded_batch.shape[1]))

                # print(encoded_batch.shape)
                feed = {rnn.inputs_: encoded_batch,
                        rnn.targets_: label_encoder.transform([batch_y]),
                        rnn.keep_prob: 0.80,
                        rnn.initial_state: state}

                if batch_counter in sampling_list:
                    network_prediction = sess.run([prediction],
                                                  feed_dict=feed)
                    print("validation =======> network_prediction: {}".format(network_prediction[0][0]),
                          "and ground truth: {}".format(batch_y-1))
                    # print(network_prediction[0])
                    # print(batch_y-1)
                    if network_prediction[0][0] == batch_y-1:
                        validation_accuracy += 1

                else:
                    batch_loss, state, _ = sess.run([rnn.loss, rnn.final_state, rnn.optimizer], feed_dict=feed)

                    total_loss += batch_loss

                    print("Epoch: {}/{}".format(e, epochs),
                          "Video Number: {}".format(batch_counter),
                          "Batch Loss: {:.3f}".format(batch_loss))
                    iteration += 1

                batch_counter += 1
                loop_counter += 1

                if batch_counter == 420:
                    total_loss = total_loss / 420
                    end_time = time.time()
                    print("===========================================================================================")
                    print("Epoch Number", e, "has ended in", end_time - start_time, "seconds for", batch_counter,
                          "videos",
                          "total loss is = {:.3f}".format(total_loss),
                          "validation accuracy is = {}".format(100*(validation_accuracy / len(sampling_list))))
                    print("===========================================================================================")
                    feed = {val_acc: validation_accuracy/len(sampling_list), tot_loss: total_loss}
                    summary = sess.run(merged_summary_op, feed_dict=feed)
                    summary_writer.add_summary(summary, e)

                    break

            if e % 30 == 0:
                print("################################################")
                print("saving the model at epoch", checkpoint_number + loop_counter)
                print("################################################")

                saver.save(sess, os.path.join(logs_path, 'lstm_loop_count_{}.ckpt'
                                              .format(checkpoint_number + loop_counter)))

    print("Run the command line:\n--> tensorboard --logdir={}".format(logs_path),
          "\nThen open http://0.0.0.0:6006/ into your web browser")

    rnn.process_node_names()
    utility.freeze_model(sess, logs_path, tf.train.latest_checkpoint(logs_path),
                         rnn, "lstm_train.pb", cs.LSTM_FREEZED_PB_NAME)

    sess.close()


if __name__ == "__main__":
    total_start_time = time.time()
    BATCH_SIZE = 1
    train()
    total_end_time = time.time()
    print("===================================================")
    print("Total Execution Time =", total_end_time - total_start_time)
    print("===================================================")

    # 1748.5977