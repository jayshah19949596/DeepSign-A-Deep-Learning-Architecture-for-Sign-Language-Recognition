import os
import cv2
import imageio
import numpy as np
import tensorflow as tf
import utils.constants as cs
import matplotlib.pyplot as plt
from utils import cv_utils, os_utils
from moviepy.editor import VideoFileClip
from tensorflow.python.tools import freeze_graph


def freeze_model(sess, logs_path, latest_checkpoint, model, pb_file_name, freeze_pb_file_name):
    print("logs_path =", logs_path)
    tf.train.write_graph(sess.graph.as_graph_def(), logs_path, pb_file_name)
    input_graph_path = os.path.join(logs_path, pb_file_name)
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = latest_checkpoint
    output_graph_path = os.path.join(logs_path, freeze_pb_file_name)
    clear_devices = False
    output_node_names = ",".join(model.nodes)
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    initializer_nodes = ""
    freeze_graph.freeze_graph(input_graph_path,
                              input_saver_def_path,
                              input_binary,
                              input_checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_path,
                              clear_devices,
                              initializer_nodes)


def prepare_batch_frames(video_path):
    fg_bg = cv2.createBackgroundSubtractorMOG2()
    video = imageio.get_reader(video_path, 'ffmpeg')
    frame_batch = np.zeros((240, 240))
    frame_batch = frame_batch.reshape((1, 240, 240))

    for i in range(len(video)):
        frame = video.get_data(i)
        edged_image = cv_utils.apply_canny(frame, 50, 150)
        rect_pts = detect_person(frame)
        fg_mask = fg_bg.apply(frame)
        fg_mask = fg_mask[int(rect_pts[0]): int(rect_pts[2]-120), int(rect_pts[1]): int(rect_pts[3]-50)]
        edged_image = edged_image[int(rect_pts[0]): int(rect_pts[2]-120), int(rect_pts[1]): int(rect_pts[3]-50)]
        fg_mask[fg_mask > 0] = 255.0
        print(fg_mask.shape)
        fg_mask = cv2.addWeighted(fg_mask, 1, edged_image, 1, 0)
        # fg_mask = cv2.bitwise_and(fg_mask, edged_image)
        reshaped_img = cv_utils.resize(fg_mask, (240, 240))
        reshaped_img = reshaped_img / 255.0
        cv2.imshow("bg_subtraction", reshaped_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        reshaped_img = reshaped_img.reshape((1, 240, 240))
        frame_batch = np.vstack((frame_batch, reshaped_img))

    frame_batch = frame_batch.reshape(frame_batch.shape[0], 240, 240, 1)
    frame_batch = frame_batch[2:, :, :, :]

    return frame_batch


def load_a_frozen_model(path_to_ckpt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
    """Transforms the box masks back to full image masks.

    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.

    Args:
      box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.

    Returns:
      A tf.float32 tensor of size [num_masks, image_height, image_width].
    """
    # TODO: Make this a public function.
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
        boxes = tf.reshape(boxes, [-1, 2, 2])
        min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
        max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
        transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
        return tf.reshape(transformed_boxes, [-1, 4])

    box_masks = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    image_masks = tf.image.crop_and_resize(image=box_masks,
                                           boxes=reverse_boxes,
                                           box_ind=tf.range(num_boxes),
                                           crop_size=[image_height, image_width],
                                           extrapolation_value=0.0)
    return tf.squeeze(image_masks, axis=3)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def detect_person(image):
    path_to_ckpt = cs.BASE_LOG_PATH+cs.MODEL_SSD+cs.OBJ_DET__PB_NAME
    output_dict = run_inference_for_single_image(image, load_a_frozen_model(path_to_ckpt))
    boxes = output_dict['detection_boxes']
    rectangle_pts = boxes[0, :] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
    # image = image[int(rectangle_pts[0]): int(rectangle_pts[2]), int(rectangle_pts[1]): int(rectangle_pts[3])]
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image[int(rectangle_pts[0]): int(rectangle_pts[2]), int(rectangle_pts[1]): int(rectangle_pts[3])])
    # plt.show()
    return rectangle_pts


def process_image(image):
    edged_image = cv_utils.apply_canny(image, 50, 150)
    rect_pts = detect_person(image)
    fg_mask = fg_bg.apply(image)
    fg_mask = fg_mask[int(rect_pts[0]): int(rect_pts[2] - 120), int(rect_pts[1]): int(rect_pts[3] - 50)]
    edged_image = edged_image[int(rect_pts[0]): int(rect_pts[2] - 120), int(rect_pts[1]): int(rect_pts[3] - 50)]
    fg_mask[fg_mask > 0] = 255.0
    # print(fg_mask.shape)
    fg_mask = cv2.addWeighted(fg_mask, 1, edged_image, 1, 0)
    reshaped_img = cv_utils.resize(fg_mask, (500, 500))
    reshaped_img = np.dstack((reshaped_img, np.zeros_like(reshaped_img), np.zeros_like(reshaped_img)))
    # cv2.imshow("bg_subtraction", reshaped_img)
    return reshaped_img


def write_videos(video_path, sub_str_1, sub_str_2):
    write_op = video_path.replace(sub_str_1, sub_str_2)
    raw_clip = VideoFileClip(video_path)
    bg_clip = raw_clip.fl_image(process_image)  # NOTE: this function expects color images!!
    bg_clip.write_videofile(write_op, audio=False)


def read_video(video_path):
    video = imageio.get_reader(video_path, 'ffmpeg')
    for i in range(len(video)):
        frame = video.get_data(i)
        detect_person(frame)


if __name__ == '__main__':
    fg_bg = cv2.createBackgroundSubtractorMOG2()
    IMAGE_SIZE = (12, 8)

    path_gen = os_utils.iterate_data(cs.BASE_DATA_PATH + cs.DATA_TRAIN_VIDEOS, ".mp4")

    for path in path_gen:
        write_videos(path, cs.DATA_TRAIN_VIDEOS, cs.DATA_BG_TRAIN_VIDEO)

    path_gen = os_utils.iterate_test_data(cs.BASE_DATA_PATH + cs.DATA_TEST_VIDEOS, ".mp4")
    for path in path_gen:
        write_videos(path, cs.DATA_TEST_VIDEOS, cs.DATA_BG_TEST_VIDEO)


