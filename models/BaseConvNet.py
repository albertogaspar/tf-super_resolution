""" Base class for Convolutional Neural Network """

import tensorflow as tf
from functools import reduce
from tensorflow.contrib import slim, keras

FLAGS = tf.app.flags.FLAGS


class BaseConvNet:
    NUM_CLASSES = 0
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0

    def __init__(self, use_pretrained_model=False):
        self.use_pretrained_model = use_pretrained_model
        self.reuse = False

    def train2(self, total_loss, global_step):
        """
        Train model.
        Args:
          total_loss: Total loss from loss().
          global_step: Integer Variable counting the number of training steps
            processed.
        Returns:
          train_op: operation for training.
        """
        pass

    def fit(self, images, train=True):
        """
        Create model and perform one (forward) step
        :param images: tf.Tensor
            4D input [batch_size, height, width, depth]
        :param train: bool
            True for training, False for testing
        :return: tf.Tensor
            2D Logits (unnormalized class probabilities) [batch_size, NUM_CLASSES]
        """
        pass


    def conv_layer(self, name, input, filters, filter_size, strides, activation=tf.nn.relu):
        """
        2D Convolutional layer
        :param name: str
            name of the layer
        :param input: tf.Tensor
            the input tensor
        :param filters: int
            the number of activation maps (i.e. out_channels)
        :param filter_size: list
            the dimension of the filter (kernel) as [height, width]
        :param strides: list
            how much to slide the kernel in each dimensions: [batch, height, width, in_channels]
        :return: tf.Tensor
            the output tensor
        """
        with tf.variable_scope(name, reuse=self.reuse):
            n_in = input.get_shape().as_list()[-1]
            if self.use_pretrained_model:
                kernel = tf.constant(self.pretrained_weights[name]['weights'], name='weights', dtype=tf.float32)
                biases = tf.constant(self.pretrained_weights[name]['biases'], name='biases', dtype=tf.float32)
            else:
                kernel = self.get_variable_weight_decay(name='weights',
                                                    shape=[filter_size[0], filter_size[1], n_in, filters],
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    wd=self.wd)
                biases = self.get_variable('biases', shape=[filters], weights_initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(input, kernel, strides=strides, padding='SAME')
            conv_with_bias = tf.nn.bias_add(conv, biases)
            if activation:
                conv_with_bias = activation(conv_with_bias)
            return conv_with_bias

    # def conv_layer(self, name, batch_input, filters=64, filter_size=3, strides=[1,1,1,1], activation=tf.nn.relu):
    #     # kernel: An integer specifying the width and height of the 2D convolution window
    #     with tf.variable_scope(name):
    #             return slim.conv2d(batch_input, filters, filter_size, strides[1], 'SAME', data_format='NHWC',
    #                                activation_fn=activation, weights_initializer=tf.contrib.layers.xavier_initializer())

    # Define our tensorflow version PRelu
    def prelu_tf(self, inputs, name='Prelu'):
        with tf.variable_scope(name):
            alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(),
                                     dtype=tf.float32)
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs - abs(inputs)) * 0.5

        return pos + neg

    # Our dense layer
    # def dense_layer(self, name, input, units=4096, activation=tf.nn.relu):
    #     output = tf.layers.dense(input, units, activation=activation,
    #                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    #     return output

    def pool_layer(self, name, input, pool_size, strides, padding='SAME', type='max'):
        """
        Pooling layer (max or avg)
        :param name: str
            name of the layer
        :param input: tf.Tensor
            the input tensor
        :param pool_size: list
            the dimension of the filter (kernel) as [height, width]
        :param strides: list
            how much to slide the kernel in each dimensions: [batch, height, width, in_channels]
        :param type: str
            max or avg
        :return: tf.Tensor
            the output tensor
        """
        if type == 'max':
            pool_fun = tf.nn.max_pool
        elif type == 'avg':
            pool_fun = tf.nn.avg_pool
        else:
            raise NotImplementedError('{0} not a valid poooling function.'.format(type))
        return pool_fun(input, [1, pool_size[0], pool_size[1], 1], strides=strides, padding=padding, name=name)

    def dense_layer(self, name, input, units=4096, activation=tf.nn.relu):
        """
        Fully connected layer.
        Flatten the given input (works in 1D)
        :param name: str
            name of the layer
        :param input: tf.Tensor
            the input tensor
        :param units: int
            the number of out_channels
        :param activation:
            a function to be used as activation (e.g. tf.nn.relu, tf.nn.softmax,...)
        :return: tf.Tensor
            the ouput tensor
        """
        with tf.variable_scope(name, reuse=self.reuse):
            # reshape 4D input [batch_size, height, width, depth] to [batch_size, height*width*depth]
            input = tf.reshape(input, [input.get_shape().as_list()[0], -1])
            n_in = input.get_shape().as_list()[1]
            if self.use_pretrained_model and name in self.pretrained_weights.keys():
                weights = tf.constant(self.pretrained_weights[name]['weights'], name='weights', dtype=tf.float32)
                biases = tf.constant(self.pretrained_weights[name]['biases'], name='biases', dtype=tf.float32)
            else:
                weights = self.get_variable_weight_decay(name='weights',
                                                         shape=[n_in, units],
                                                         weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                         wd=self.wd)
                biases = self.get_variable('biases', shape=[units], weights_initializer=tf.contrib.layers.xavier_initializer())
            dense = tf.nn.bias_add(tf.matmul(input, weights), biases)
            if activation:
                dense = activation(dense)
            return dense

    def batch_normalization(self, name, input, decay=0.999):
        return slim.batch_norm(input, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                               scale=False, fused=True, is_training=self.train)

    def get_variable_weight_decay(self, name, shape, weights_initializer, wd=None):
        """
        Create or retrieve a variable using tf.get_variable and add it to the dict of model variables.
        If wd is specified then the L2 loss of this variable is added to the collection 'losses'.
        :param name: str
            the variable name
        :param shape: list
            the variable dimension
        :param weights_initializer:
            the variable initializer
        :param wd: tf.float32
            weight decay rate
        :return: tf.Variable
            the variable
        """
        weights = self.get_variable(name, shape=shape, weights_initializer=weights_initializer)
        if wd:
            wd_reg = tf.multiply(tf.nn.l2_loss(weights), wd)
            if 'generator' in name:
                tf.add_to_collection('gen_losses', wd_reg) # add value to the collection
            elif 'discriminator' in name:
                tf.add_to_collection('disc_losses', wd_reg)
            else:
                tf.add_to_collection('losses', wd_reg)
        return weights

    def get_variable(self, name, shape, weights_initializer):
        """
        Create or retrieve a variable with get_variable and add it to the dict of model variables
        :param name: str
            the variable name
        :param shape: list
            the variable dimension
        :param weights_initializer:
            the variable initializer
        :return: tf.Tensor
            the variable
        """
        weights = tf.get_variable(name, shape=shape, initializer=weights_initializer)
        return weights

    def predict(self, images):
        logits = self.fit(images)
        probs = tf.nn.softmax(logits)
        predictions = tf.argmax(probs, axis=1)
        return logits, predictions
