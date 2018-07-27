import tensorflow as tf
import numpy as np
from data_loader import convert_back
from models.BaseConvNet import BaseConvNet
from models.VGG19 import VGG19
from utils import psnr

FLAGS = tf.app.flags.FLAGS


class Generator(BaseConvNet):
    """
    Generator of super resultion (SR) images (4x upscaling)
    """

    def __init__(self, wd=None, use_pretrained_model=False):
        super().__init__(use_pretrained_model)
        self.wd = wd
        if FLAGS.load_vgg:
            self.vgg19 = VGG19()

    def train2(self, total_loss, global_step, gs_update=True):
        """
        Train model.(adapted from)
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        Args:
          total_loss: Total loss from loss().
          global_step: Integer Variable counting the number of training steps
            processed.
        Returns:
          train_op: operation for training.
        """
        with tf.variable_scope('generator_train'):
            # Define the learning rate and global step
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 400000,
                                                       0.1, staircase=False)

            opt = tf.train.AdamOptimizer(learning_rate)
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            if gs_update:
                gen_train_op = opt.minimize(total_loss, global_step, var_list=vars)
            else:
                gen_train_op = opt.minimize(total_loss, None, var_list=vars)
        return global_step, gen_train_op

    def train_wgan(self, total_loss, global_step, gs_update=True):
        """
        Train model.(adapted from)
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        Args:
          total_loss: Total loss from loss().
          global_step: Integer Variable counting the number of training steps
            processed.
        Returns:
          train_op: operation for training.
        """
        with tf.variable_scope('generator_train'):
            # Define the learning rate and global step
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 400000,
                                                       0.1, staircase=False)

            opt = tf.train.RMSPropOptimizer(learning_rate)
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            if gs_update:
                gen_train_op = opt.minimize(total_loss, global_step, var_list=vars)
            else:
                gen_train_op = opt.minimize(total_loss, None, var_list=vars)
        return global_step, gen_train_op

    def prelu(self, input, alpha=0.2):
        """
        Probabilistic ReLu with fixed alpha
        """
        with tf.variable_scope('PReLu'):
            alphas = tf.get_variable('alpha', input.get_shape()[-1], initializer=tf.zeros_initializer(),
                                     dtype=tf.float32)
        pos = tf.nn.relu(input)
        neg = alphas * (input - abs(input)) * 0.5

        return pos + neg

    def fit(self, images, B_blocks=16, S_blocks=2, train=True, reuse=False):
        """
        One step of training for the Generator
        :param images: tf.Tensor
            4D [batch_size, height, width, in_channels]
        :param train: bool
        :param B_blocks: int
            Number of residula blocks
        :param S_blocks:
            Number of shuffling blocks
        :return: tf.Tensor
            SR image
        """
        with tf.variable_scope('generator'):
            self.train = train
            self.reuse = reuse

            # Conv + PReLu
            conv1 = self.conv_layer('conv1', images, filters=64, filter_size=[9, 9], strides=[1,1,1,1],
                                      activation=self.prelu)

            # B residual blocks
            residual_block = conv1
            for i in range(1, B_blocks+1):
                residual_block = self.residual_block('res{0}'.format(i), residual_block, kernel_size=[3,3], out_filters=64)

            # Conv + BN + Elementwise sum
            conv2 = self.conv_layer('conv2', residual_block, filters=64, filter_size=[3, 3], strides=[1,1,1,1],
                                      activation=None)
            conv2 = self.batch_normalization('conv2/bn', conv2)
            b_sum = conv1 + conv2  # elementwise addition for each feature map

            # SubPixel Conv Layer (LR -> HR): Conv + Shuffler + PReLu
            subpixel_conv = b_sum
            for i in range(1, S_blocks+1):
                subpixel_conv = self.sub_pixel_conv('subpixel_conv{0}'.format(i), subpixel_conv, r=2)

            # Final Conv
            conv4 = self.conv_layer('final', subpixel_conv, filters=3, filter_size=[9,9],
                                    strides=[1, 1, 1, 1], activation=None)

            if not FLAGS.inference:
                images_dim = images.get_shape().as_list()
                conv4.set_shape([images_dim[0], images_dim[1]*4, images_dim[2]*4, images_dim[3]])
            return conv4

    def residual_block(self, name, input, kernel_size=[3, 3], out_filters=64):
        """
        Given an input X, a residul block is  composed of several stacked convolutioal layers
        and an identity shortcut connection that start at X skips all this layers.
        The ouput is obtained adding (element-wise add) X to the output of the concolutinal layers.
        Notice that identity shortcut connections add neither extra parameter nor
        computational complexity to the model.

        This Residual Block is implemnted following the sequnce:
        Conv - BN - ReLU - Conv - BN - Addition - Relu (Original paper)
        :param name: str
        :param input: tf.Tensor
            4D input [batch_size, height, width, n_channels]
        :param kernel_size: list
            Kernel dimension
        :param out_filters: int
            Number of filters in the convolution layers.
            For now this is forced to be equals to n_channels of X. It is a parameter beacuse I plan to
            support different dimensions in the future.
        :return:
        """
        with tf.variable_scope(name):
            in_filters = input.get_shape().as_list()[-1]
            if in_filters == out_filters:
                strides = [1,1,1,1]
            elif in_filters*2 == out_filters:
                strides = [1,2,2,1]
            else:
                raise ValueError('From one block to an other the number of filter should either remain the same or double.'
                                 'No other option is available')

            fx = self.conv_layer('conv1', input, out_filters,
                                 kernel_size, strides, activation=self.prelu)
            fx = self.batch_normalization('conv1/bn', fx)
            fx = self.prelu(fx)
            fx = self.conv_layer('conv2', fx, out_filters,
                                 kernel_size, strides, activation=None)
            fx = self.batch_normalization('conv2/bn', fx)
            # If x and F has different dimensions we zero pad the input to make them match
            if in_filters != out_filters:
                pooled_input = self.pool_layer('avg_pool', input, pool_size=[2, 2],
                                               strides=strides, padding='VALID', type='avg')
                # pad only the last dimesions (channels) inserting (in total) a numbe of 0s equal
                # to the number of filters (i.e. we double the channels)
                input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [in_filters // 2, in_filters // 2]])
            y = tf.add(fx, input)  # elementwise addition for each feature map
            # y = self.prelu(y)
            return y

    def sub_pixel_conv(self, name, input, r=2):
        """
        Sub-pixel convolution layer that aggregates the feature maps from LR space and
        builds the SR image in a single step.
        Not that this operation is applicable when mod(n_filters, r**2) = 0
        (e.g input.get_shape() = [16,4,4,32], r=4 -> 32 % 4**2 = 0)
        :param input: tf.Tensor
            tensor shape [batch_size, height, width, n_filters]
        :param r: int
            upsampling scale
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv_layer('conv', input, filter_size=[3, 3], filters=256,
                                   strides=[1, 1, 1, 1], activation=None)
            conv_shuff = self.pixelShuffler(conv, r)
            out = self.prelu(conv_shuff)
            return out

    def pixelShuffler(self, inputs, scale=2):
        size = tf.shape(inputs)
        batch_size = size[0]
        h = size[1]
        w = size[2]
        c = inputs.get_shape().as_list()[-1]

        # Get the target channel size
        channel_target = c // (scale * scale)
        channel_factor = c // channel_target

        shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
        shape_2 = [batch_size, h * scale, w * scale, 1]

        # Reshape and transpose for periodic shuffling for each channel
        input_split = tf.split(inputs, channel_target, axis=3)
        output = tf.concat([self.phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

        return output

    def phaseShift(self, inputs, scale, shape_1, shape_2):
        # Tackle the condition when the batch is None
        X = tf.reshape(inputs, shape_1)
        X = tf.transpose(X, [0, 1, 3, 2, 4])

        return tf.reshape(X, shape_2)

    def periodic_shuffling(self, input, r):
        input_shape = input.get_shape().as_list()
        expected_out_shape = [input_shape[0], # batch_size
                              input_shape[1] * r, # height
                              input_shape[2] * r, # width
                              int(input_shape[3] / (r ** 2))] # n_channels

        out = tf.depth_to_space(input, r)
        assert out.get_shape() == expected_out_shape
        return out

    def pixelwise_mse_loss(self, hr_image, sr_image):
        """
        Pixel wise MSE loss
        :param hr_image: tf.Tensor
            Natural Image
            [batch_size, height, width, n_channels]
        :param sr_image:
            Image produced by the Generator network
            [batch_size, height, width, n_channels]
        :return:
            Loss tensor of of type float
        """
        with tf.variable_scope('mse_loss'):
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(hr_image - sr_image), axis=[3]), name='pixelwise_mse')
            # tf.add_to_collection('gen_losses', loss)
            return loss

    def psnr(self, hr_image, sr_image):
        # Peak-signal-to-noise-ratio
        _psnr = psnr(hr_image, sr_image)
        # tf.summary.scalar('psnr', _psnr)
        return _psnr

    def vgg_loss(self, hr_image, sr_image, embeddings='VGG54'):
        """
        Euclidean distance between the feature representations of a SR image (by generator network)
        and the reference natural HR image.
        :param hr_image: tf.Tensor
            Natural Image
            [batch_size, height, width, n_channels]
        :param sr_image:
            Image produced by the Generator network
            [batch_size, height, width, n_channels]
        :param embeddings: str
            VGG22 or VGG54
        :return:
            Loss tensor of of type float
        """
        _, feature_map_sr = self.vgg19.fit(sr_image, embeddings=embeddings)
        _, feature_map_hr = self.vgg19.fit(hr_image, embeddings=embeddings, reuse=True)
        with tf.variable_scope('vgg_loss'):
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(feature_map_hr - feature_map_sr), axis=[3]), name='vgg_loss')
        # tf.add_to_collection('gen_losses', loss * FLAGS.vgg_loss_scale)
        return loss

    def adversarial_loss(self, logit_SR):
        """
        Adversarial loss
        :param logit_SR: list, array
            Probability that the image reconstructed by the generator is a natural HR image and
            not an artificial SR image.
        :return: tf.Tensor
            Loss tensor of of type float
        """
        with tf.variable_scope('adversarial_loss'):
            # minimize the probability that the discriminator is correct
            loss = tf.reduce_mean(-tf.log(logit_SR + FLAGS.eps))
            # tf.add_to_collection('gen_losses', loss * FLAGS.adversarial_loss_scale)
            return loss

    def adversarial_loss_wgan(self, logit_SR):
        """
        Adversarial loss
        :param logit_SR: list, array
            Probability that the image reconstructed by the generator is a natural HR image and
            not an artificial SR image.
        :return: tf.Tensor
            Loss tensor of of type float
        """
        # minimize the probability that the discriminator is correct
        # create imgs to which D will assign high values (max logit_SR = min -logit_SR)
        loss = tf.reduce_mean(-logit_SR)
        # tf.add_to_collection('gen_losses', loss * FLAGS.adversarial_loss_scale)
        return loss

    def total_loss(self):
        # The total loss is defined as the sum of all the considered losses (content, adversarial)
        # plus all of the weight decay terms (L2 loss).
        return tf.add_n(tf.get_collection('gen_losses'), name='total_loss')

