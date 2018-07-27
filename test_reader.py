from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_loader import *
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 96,
                           "Cropped image size")
tf.app.flags.DEFINE_string('log_dir', './log',
                           "Directory where the train data are stored.")
tf.app.flags.DEFINE_string('train_data_dir_HR', './data/RAISE_HR',
                           "Directory where the train data are stored.")
tf.app.flags.DEFINE_string('train_data_dir_LR', './data/RAISE_LR',
                           "Directory where the train data are stored.")
tf.app.flags.DEFINE_integer('batch_size', 8,
                            "How many example to process in each batch")


def input(argv=None):
    imgs_LR, imgs_HR = inputs(False, FLAGS.batch_size)
    # Merge all summary inforation.
    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        # Create a writer for the summary data.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        imgs_LR, imgs_HR, summ_str = sess.run([imgs_LR, imgs_HR, summary])
        summary_writer.add_summary(summ_str, 1)
        summary_writer.flush()
        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MkDir(FLAGS.log_dir)
    input()


if __name__ == '__main__':
    tf.app.run()
