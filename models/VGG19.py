import tensorflow as tf
from tensorflow.contrib import slim


class VGG19:

    def __init__(self, wd=0.0005):
        self.wd = wd

    def fit(self, images, embeddings='VGG54', num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
        with tf.variable_scope('vgg_19', 'vgg_19', [images], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
        # with tf.variable_scope('vgg_19', reuse=reuse):
        #     with slim.arg_scope([slim.conv2d, slim.fully_connected],
        #                         activation_fn=tf.nn.relu,
        #                         weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        #                         weights_regularizer=slim.l2_regularizer(0.0005),
        #                         trainable=False):
                emb = None
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                if embeddings == 'VGG22':
                    emb = net
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
                if embeddings == 'VGG54':
                    emb = net
                net = slim.max_pool2d(net, [2, 2], scope='pool5')
            if emb is not None:
                return net, emb
            else:
                return net