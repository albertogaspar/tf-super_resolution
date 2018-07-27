from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def convert_back(img, LR=True):
    if LR:
        img *= 255
    else:
        img = ((img + 1) / 2) * 255
    return img

def load_and_normalize(filename, LR=True):
    with tf.name_scope('load_img'):
        img_string = tf.read_file(filename)

        img_decoded = tf.image.decode_png(img_string, channels=3)

    # Normalize img to [0,1]
    with tf.name_scope('normalize_img'):
        # img = img_decoded / 255.
        img = tf.image.convert_image_dtype(img_decoded, dtype=tf.float32)

        with tf.control_dependencies([img]):
            # tf.identity can be used used as a dummy node to update a reference to the tensor
            img = tf.identity(img)

        # Normalize LR imgs to [0,1], HR imgs to [-1,1]
        if not LR:
            img = (img * 2) - 1

    return img


def preprocess_imgs(filename_LR, filename_HR, distort=False):
    # load and preprocess the image
    img_LR = load_and_normalize(filename_LR, LR=True)
    img_HR = load_and_normalize(filename_HR, LR=False)

    with tf.name_scope('crop_img'):
        if FLAGS.mode == 'train':
            # crop images for training [IMG_SIZE, IMG_SIZE, 3]
            input_size = tf.shape(img_LR)
            offset_w = tf.cast(
                tf.floor(tf.random_uniform([], 0, tf.cast(input_size[1], tf.float32) - FLAGS.image_size)),
                dtype=tf.int32)
            offset_h = tf.cast(
                tf.floor(tf.random_uniform([], 0, tf.cast(input_size[0], tf.float32) - FLAGS.image_size)),
                dtype=tf.int32)
            img_LR = tf.image.crop_to_bounding_box(img_LR, offset_h, offset_w, FLAGS.image_size,
                                                   FLAGS.image_size)
            img_HR = tf.image.crop_to_bounding_box(img_HR, offset_h * 4, offset_w * 4, FLAGS.image_size * 4,
                                                    FLAGS.image_size * 4)

    img_LR = tf.identity(img_LR)
    img_HR = tf.identity(img_HR)

    return img_LR, img_HR


def _generate_image_and_label_batch(image_LR, image_HR, min_queue_examples,batch_size, shuffle):
    # Create a queue that enqueue at some point the image_LR, image_HR.
    # It shuffles the examples, and then read 'batch_size' [images_LR, images_HR] from the queue.
        num_preprocess_threads = 4
        if shuffle:
            images_LR_batch, images_HR_batch = tf.train.shuffle_batch(
                [image_LR, image_HR],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                # Tensorflow docs suggest capacity = min_after_dequeue + (threads+1) * batch_size
                capacity=min_queue_examples + 4 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images_LR_batch, images_HR_batch = tf.train.batch(
                [image_LR, image_HR],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 4 * batch_size)

        images_LR_batch.set_shape([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
        images_HR_batch.set_shape([FLAGS.batch_size, FLAGS.image_size*4, FLAGS.image_size*4, 3])

        return images_LR_batch, images_HR_batch


def inputs(eval_data, batch_size):
    """Construct input.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      batch_size: Number of images per batch.

    Returns:
      images_LR: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      images_HR: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    """
    if not eval_data:
        imgs_LR_paths = [os.path.join(FLAGS.train_data_dir_LR, fn) for fn in os.listdir(FLAGS.train_data_dir_LR)]
        imgs_HR_paths = [os.path.join(FLAGS.train_data_dir_HR, fn) for fn in os.listdir(FLAGS.train_data_dir_HR)]
    else:
        imgs_LR_paths = [os.path.join(FLAGS.test_data_dir_LR, fn) for fn in os.listdir(FLAGS.test_data_dir_LR)]
        imgs_HR_paths = [os.path.join(FLAGS.test_data_dir_HR, fn) for fn in os.listdir(FLAGS.test_data_dir_HR)]

    samples_per_epoch = np.ceil(len(imgs_LR_paths) / FLAGS.batch_size)
    imgs_LR_paths = tf.convert_to_tensor(imgs_LR_paths, tf.string)
    imgs_HR_paths = tf.convert_to_tensor(imgs_HR_paths, tf.string)

    input_queue = tf.train.slice_input_producer((imgs_LR_paths, imgs_HR_paths), capacity=4096)

    img_LR, img_HR = preprocess_imgs(input_queue[0], input_queue[1])

    min_queue_examples = 4096

    if FLAGS.mode == 'train':
        img_LR.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
        img_HR.set_shape([FLAGS.image_size*4, FLAGS.image_size*4, 3])

    # Generate a batch of images_LR, images_HR by building up a queue of examples.
    return _generate_image_and_label_batch(img_LR, img_HR, min_queue_examples, batch_size,shuffle=False)

