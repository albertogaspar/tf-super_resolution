import tensorflow as tf
import numpy as np
from functools import reduce
from models.BaseConvNet import BaseConvNet

FLAGS = tf.app.flags.FLAGS

class Discriminator(BaseConvNet):
    """
    Discriminator: trained to distinguidh real high resilution (HR) images from the SR images produced
    by the generator network.
    """

    def __init__(self, wd=None, use_pretrained_model=False):
        super().__init__(use_pretrained_model)
        self.wd = wd

    def train2(self, total_loss, global_step):
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
        with tf.variable_scope('discriminator_train'):
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 400000,
                                                       0.1, staircase=False)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dicriminator')
            disc_train_op = opt.minimize(total_loss, global_step, var_list=vars)
        return global_step, disc_train_op

    def train_wgan(self, total_loss, global_step, c=None):
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
        with tf.variable_scope('discriminator_train'):
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 400000,
                                                       0.1, staircase=False)

            # Apply gradients and clip them
            opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dicriminator')
            disc_train_op = opt.minimize(total_loss, global_step, var_list=vars)
            clip_D = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in vars]
            # grads = opt.compute_gradients(total_loss)
            # clipped_grads = [(tf.clip_by_value(grad, -c, c), var) for grad, var in grads]
            # disc_train_op = opt.apply_gradients(clipped_grads)
            # global_step = tf.assign(global_step, global_step+1)
        return global_step, tf.group(disc_train_op, clip_D)

    def prelu(self, x, alpha=0.2):
        """
        Probabilistic ReLu with fixed alpha
        """
        return tf.nn.leaky_relu(x, alpha=alpha)

    def fit(self, images, train=True, reuse=False):
        """
        One step of training of the VGG 16 model
        :param images: tf.Tensor
            4D [batch_size, height, width, in_channels],
            for CIFAR 10 input is [batch_size, 32, 32, 3]
            for Imagenet input is assumed to be [batch_size, 224, 224, 3]
            Input is assumed to be normalized (e.g. (x-mean)/(std + eps))
        :return: tf.Tensor
            2D probabilities [batch_size, 2] obtained from the last layer of the network (sigmoid).
            (i.e. probabilities of image being HR or SR)
        """
        self.train = train
        self.reuse = reuse
        with tf.variable_scope('dicriminator', reuse=reuse):
            block = self.conv_layer('block0/conv', images, filters=64, strides=[1,1,1,1],
                                    filter_size=[3, 3], activation=lambda x : tf.nn.leaky_relu(x, alpha=0.2))

            block = self.block('block1', block, filters=64, strides=[1,2,2,1])

            block = self.block('block2', block, filters=128, strides=[1, 1, 1, 1])
            block = self.block('block3', block, filters=128, strides=[1, 2, 2, 1])

            block = self.block('block4', block, filters=256, strides=[1, 1, 1, 1])
            block = self.block('block5', block, filters=256, strides=[1, 2, 2, 1])

            block = self.block('block6', block, filters=512, strides=[1, 1, 1, 1])
            block = self.block('block7', block, filters=512, strides=[1, 2, 2, 1])

            # Dense layers -- out: [batch_size, NUM_CLASSES]
            dense1 = self.dense_layer('dense_1', block, units=1024,
                                      activation=lambda x : tf.nn.leaky_relu(x, alpha=0.2))
            logit_true = self.dense_layer('dense_2', dense1, units=1, activation=None)
            prob_true = tf.nn.sigmoid(logit_true)
            return logit_true, prob_true

    def block(self, name, input, filters, strides):
        with tf.variable_scope('{0}'.format(name), reuse=self.reuse):
            block = self.conv_layer('conv'.format(name), input, filters=filters, strides=strides,
                                filter_size=[3, 3], activation=None)
            block = self.batch_normalization('bn'.format(name), block)
        block = tf.nn.leaky_relu(block, alpha=0.2)
        return block

    def adversarial_loss(self, logit_HR, logit_SR):
        """
        Adversarial loss
        :param probab:
            Probability that the image reconstructed by the generator is a natural HR image and
            not an artificial SR image.
        :return: tf.Tensor
            Loss tensor of of type float
        """
        with tf.variable_scope('disc_adv_loss'):
            # use one-sided label smoothing on true imgs only (I.Goodfellow, NIPS 2016 Tutorial: Generative Adversarial Networks)
            # loss_HR = -0.9 * tf.log(logit_HR + FLAGS.eps) - 0.1 * tf.log(1-logit_HR + FLAGS.eps)
            loss_HR = -tf.log(logit_HR + FLAGS.eps)
            tf.summary.scalar('discriminator_loss_HR', tf.reduce_mean(loss_HR))
            loss_SR = -tf.log(1-logit_SR + FLAGS.eps)
            tf.summary.scalar('discriminator_loss_SR', tf.reduce_mean(loss_SR))
            tf.nn.sigmoid_cross_entropy_with_logits
            loss = tf.reduce_mean(loss_HR+loss_SR)
            # tf.add_to_collection('disc_losses', loss)
            return loss

    def adversarial_loss_wgan(self, logit_HR, logit_SR):
        """
        Adversarial loss for WGAN (no log)
        :param probab:
            Probability that the image reconstructed by the generator is a natural HR image and
            not an artificial SR image.
        :return: tf.Tensor
            Loss tensor of of type float
        """
        # use one-sided label smoothing (I.Goodfellow, NIPS 2016 Tutorial: Generative Adversarial Networks)
        loss_HR = tf.reduce_mean(logit_HR) # assign high values to real images
        tf.summary.scalar('discriminator_loss_HR', loss_HR)
        loss_SR = tf.reduce_mean(logit_SR) # assign low values to fake images
        tf.summary.scalar('discriminator_loss_SR', loss_SR)
        loss = tf.reduce_mean(logit_SR - logit_HR) # minimize wasserstein distance
        tf.summary.scalar('discriminator_loss_sum', loss_SR)
        # tf.add_to_collection('disc_losses', loss)
        return loss


    def total_loss(self):
        # The total loss is defined as the sum of all the considered losses (content, adversarial)
        # plus all of the weight decay terms (L2 loss).
        return tf.add_n(tf.get_collection('disc_losses'), name='total_loss')