from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_loader import *
from models.Discriminator import Discriminator
from models.Generator import Generator
import tensorflow as tf
import time


FLAGS = tf.app.flags.FLAGS

### Files and Folders ###
tf.app.flags.DEFINE_string('pretrained_models', './pretrained_models/',
                           "Directory where the weights for the pretrained models are stored.")
tf.app.flags.DEFINE_string('weights_dir', './weights/',
                           "Directory where the weights for the pretrained models are stored.")
tf.app.flags.DEFINE_string('log_dir', './logs/',
                           "Directory where the summaries are stored.")
tf.app.flags.DEFINE_string('checkpoint_dir', './ckpts/',
                           "Directory where the checkpoints are stored.")

tf.app.flags.DEFINE_string('train_data_dir_HR', './data/RAISE_HR/',
                           "Directory where the train data are stored.")
tf.app.flags.DEFINE_string('train_data_dir_LR', './data/RAISE_LR/',
                           "Directory where the train data are stored.")
tf.app.flags.DEFINE_string('test_data_dir_HR', './data/RAISE_HR_inf/',
                           "Directory where the train data are stored.")
tf.app.flags.DEFINE_string('test_data_dir_LR', './data/RAISE_LR_inf/',
                           "Directory where the train data are stored.")
tf.app.flags.DEFINE_string('test_img', './data/RAISE_LR_inf/img_005.png',
                           "Directory where the train data are stored.")
### Pretrained models flags ###
tf.app.flags.DEFINE_bool('load_vgg', False,
                         'Load pretrained weights for VGG 19')
tf.app.flags.DEFINE_bool('load_gen', True,
                         'Load pretrained weights for Generator')
tf.app.flags.DEFINE_bool('load_disc', True,
                         'Load pretrained weights for Discriminator')
### Model hyper- parameters ###
tf.app.flags.DEFINE_integer('batch_size', 8,
                            "How many example to process in each batch")
tf.app.flags.DEFINE_integer('n_iter', 100000,
                           "Number of iterations for model's training")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            "initial learning rate")
tf.app.flags.DEFINE_float('vgg_loss_scale', 0.0061,
                            "Scaling for the vgg (content) loss in the generator loss")
tf.app.flags.DEFINE_float('adversarial_loss_scale', 1e-3,
                            "Scaling for the adversarial loss in the generator loss")
tf.app.flags.DEFINE_float('eps', 1e-12,
                            "")
### Data parameters ###
tf.app.flags.DEFINE_string('task', 'SRGAN',
                            "SRResNet or SRGAN")
tf.app.flags.DEFINE_string('mode', 'train',
                            "train, inference")
tf.app.flags.DEFINE_integer('image_size', 24,
                           "Image size")
tf.app.flags.DEFINE_integer('log_freq', 100,
                           "Frequency for logging/summary")
tf.app.flags.DEFINE_integer('ckpt_freq', 10000,
                           "Frequency for checkpoints")
tf.app.flags.DEFINE_bool('inference', False,
                         'Load pretrained weights for VGG 19')
tf.app.flags.DEFINE_bool('adjust_contrast', True,
                         'Adjust(increase) contrast at inference time')


def train_SRGAN():
    global_step = tf.train.get_or_create_global_step()

    # read input batch
    with tf.device('/cpu:0'):
        imgs_LR, imgs_HR = inputs2(False, FLAGS.batch_size)

    ##################################################
    #          GENERATOR - SR IMAGE created          #
    ##################################################
    generator = Generator()
    imgs_SR = generator.fit(imgs_LR, train=True, reuse=False)
    # variables for generator (SRResNet)
    if FLAGS.load_gen and not FLAGS.load_disc:
        variables_to_restore_srgan = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        srgan_saver = tf.train.Saver(variables_to_restore_srgan)

    # Display the images in the tensorboard.
    tf.summary.image('images_LR', tf.image.resize_images(imgs_LR, [FLAGS.image_size*4, FLAGS.image_size*4])) # Bilinear
    tf.summary.image('images_HR', imgs_HR)
    tf.summary.image('images_SR', imgs_SR)

    ###########################################
    #          DISCRIMINATOR - train          #
    ###########################################
    discriminator = Discriminator()
    with tf.name_scope('discriminator_HR'):
        logit_HR, probab_HR = discriminator.fit(imgs_HR, train=True, reuse=False)
    with tf.name_scope('discriminator_SR'):
        logit_SR, probab_SR = discriminator.fit(imgs_SR, train=True, reuse=True)
    if FLAGS.load_gen and FLAGS.load_disc:
        variables_to_restore_srgan = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator') + \
                                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dicriminator')
        srgan_saver = tf.train.Saver(variables_to_restore_srgan)
    disc_loss = discriminator.adversarial_loss(logit_HR=probab_HR, logit_SR=probab_SR)
    global_step, disc_train_op = discriminator.train2(disc_loss, global_step)

    ###########################################
    #            GENERATOR - train            #
    ###########################################
    with tf.control_dependencies([disc_train_op, disc_loss]): # ensure that disc has done one step
        adversarial_loss = generator.adversarial_loss(probab_SR)
        if FLAGS.load_vgg:
            content_loss = generator.vgg_loss(imgs_HR, imgs_SR)
            content_loss_type = 'vgg'
            gen_loss = FLAGS.vgg_loss_scale * content_loss + FLAGS.adversarial_loss_scale * adversarial_loss
        else:
            content_loss = generator.pixelwise_mse_loss(imgs_HR, imgs_SR)
            content_loss_type = 'mse'
            gen_loss = content_loss + FLAGS.adversarial_loss_scale * adversarial_loss
        _, gen_train_op = generator.train2(gen_loss, global_step, gs_update=False) # No update to gs

    # variables for VGG19
    if FLAGS.load_vgg:
        variables_to_restore_vgg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
        vgg_saver = tf.train.Saver(variables_to_restore_vgg)
    saver = tf.train.Saver()

    ###########################################
    #               SUMMARIES                 #
    ###########################################
    # Moving average on loss
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    losses_list = [disc_loss, content_loss, adversarial_loss, gen_loss]
    update_loss = exp_averager.apply(losses_list)
    disc_loss_avg, content_loss_avg, adversarial_loss_avg, gen_loss_avg = \
        [exp_averager.average(var) for var in losses_list]
    tf.summary.scalar('discriminator_loss', disc_loss_avg)
    tf.summary.scalar('gen_{0}_loss'.format(content_loss_type), content_loss_avg)
    tf.summary.scalar('gen_adversarial_loss', adversarial_loss_avg)
    tf.summary.scalar('generator_loss', gen_loss_avg)

    # Merge all summary inforation.
    summary = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Load pretrained model
        if FLAGS.load_gen and FLAGS.load_disc:
            print('Loading weights for SRGAN generator and discriminator...')
            srgan_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.pretrained_models, 'srgan')))
        elif FLAGS.load_gen:
            print('Loading weights for SRGAN generator...')
            srgan_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.pretrained_models, 'srresnet')))
        if FLAGS.load_vgg:
            print('Loading weights for VGG19..')
            vgg_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.pretrained_models, 'vgg19')))

        print('Starting training procedure...')
        start = time.time()
        for it in range(FLAGS.n_iter):
            gs, _, d_loss, _, g_loss, _, summ = sess.run([global_step, update_loss, disc_loss_avg, disc_train_op,
                                                       gen_loss_avg, gen_train_op, summary])
            if it % FLAGS.log_freq == 0 and it > 0:
                t = (time.time() - start)
                print('{0} iter, gen_loss: {1}, disc_loss: {2}, img/sec: {3}'.format(gs, g_loss, d_loss,
                                                                                     FLAGS.log_freq * FLAGS.batch_size / t))
                summary_writer.add_summary(summ, gs)
                summary_writer.flush()
                start = time.time()
            if it % FLAGS.ckpt_freq == 0 and it > 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=gs)
        coord.request_stop()
        coord.join(threads)

def train_SRResNet():
    global_step = tf.train.get_or_create_global_step()

    with tf.device('/cpu:0'):
        imgs_LR, imgs_HR = inputs2(False, FLAGS.batch_size)

    # train generator (no adv. loss)
    generator = Generator()
    imgs_SR = generator.fit(imgs_LR, train=True, reuse=False)

    # Display the training images in the visualizer.
    tf.summary.image('images_LR', imgs_LR)
    tf.summary.image('images_HR', imgs_HR)
    tf.summary.image('images_SR', imgs_SR)

    # Restore
    if FLAGS.load_gen:
        variables_to_restore_srgan = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        srgan_saver = tf.train.Saver(variables_to_restore_srgan)
    # Restore variables for VGG19 and SRResNet (pretrained model)
    if FLAGS.load_vgg:
        variables_to_restore_vgg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
        vgg_saver = tf.train.Saver(variables_to_restore_vgg)

    mse_loss = generator.pixelwise_mse_loss(imgs_HR, imgs_SR)
    psnr = generator.psnr(imgs_HR, imgs_SR)
    # vgg_loss = generator.vgg_loss(imgs_HR, imgs_SR)
    global_step, gen_train_op = generator.train2(mse_loss, global_step)

    ###########################################
    #               SUMMARIES                 #
    ###########################################
    # Moving average on loss
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([mse_loss])
    gen_loss = exp_averager.average(mse_loss)
    tf.summary.scalar('PSNR', psnr)
    tf.summary.scalar('MSE', mse_loss)
    tf.summary.scalar('generator loss', gen_loss)

    saver = tf.train.Saver()
    # Merge all summary inforation.
    summary = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        if FLAGS.load_gen:
            print('Loading pre-trained SRResNet (i.e. Generator)...')
            srgan_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.pretrained_models, 'srresnet')))
        if FLAGS.load_vgg:
            print('Loading pre-trained VGG 19...')
            vgg_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.pretrained_models, 'vgg19')))

        print('Starting training procedure...')
        start = time.time()
        for it in range(FLAGS.n_iter):
            gs, _, loss, _, summ = sess.run([global_step, update_loss, gen_loss, gen_train_op, summary])
            if it % FLAGS.log_freq == 0 and it > 0:
                t = (time.time() - start)
                print('{0} iter, loss: {1}, img/sec: {2}'.format(gs, loss, it * FLAGS.batch_size / t))
                print(t)
                summary_writer.add_summary(summ, gs)
                summary_writer.flush()
            if it % FLAGS.ckpt_freq == 0 and it > 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=gs+70000)
        coord.request_stop()
        coord.join(threads)


def train():
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
        print('Log files deleted.')
    tf.gfile.MkDir(FLAGS.log_dir)
    if tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        print('Checkpoint files deleted.')
    tf.gfile.MkDir(FLAGS.checkpoint_dir)
    if FLAGS.task == 'SRResNet':
        train_SRResNet()
    elif FLAGS.task == 'SRGAN':
        train_SRGAN()
    else:
        raise NotImplementedError

def inference():
    filepath = tf.convert_to_tensor(FLAGS.test_img, tf.string)
    imgs_LR = tf.read_file(filepath)
    imgs_LR = tf.image.decode_png(imgs_LR, channels=3)
    imgs_LR = imgs_LR / 255

    ##################################################
    #          GENERATOR - SR IMAGE created          #
    ##################################################
    generator = Generator()
    imgs_LR_ph = tf.placeholder(tf.float32, [None, None, None, 3])
    imgs_SR = generator.fit(imgs_LR_ph, train=True, reuse=False)

    # Restore
    if FLAGS.load_gen:
        variables_to_restore_srgan = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        srgan_saver = tf.train.Saver(variables_to_restore_srgan)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        if FLAGS.load_gen:
            print('Loading pre-trained Generator...')
            srgan_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.pretrained_models, 'srgan')))
        _img = sess.run(imgs_LR)
        _img = np.asarray(_img)
        sr = sess.run(imgs_SR, feed_dict={imgs_LR_ph: [_img]})
    converted_img = convert_back(sr[0], LR=False)
    filename = FLAGS.test_img.replace('.png', '_SRGAN_MSE_70k_ac.png')
    import cv2
    cv2.imwrite(filename, cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR))

def save_imgs(width, imgs, LR=True, name='src'):
    print('Saving SR image...')
    from scipy.misc import imsave
    delta = width // FLAGS.image_size
    lr = np.concatenate(imgs[0:delta], axis=1)
    for i in range(delta, len(imgs), delta):
        tmp = np.concatenate(imgs[i:i+delta], axis=1)
        try:
            lr = np.concatenate([lr, tmp], axis=0)
        except:
            continue
    converted_img = convert_back(lr, LR=LR)
    print(converted_img.shape)
    import cv2
    cv2.imwrite('{0}.png'.format(name), converted_img)
    print('image saved')

def main(argv=None):
    if FLAGS.inference:
        print('INFERENCE mode')
        inference()
    else:
        print('TRAIN mode')
        train()

if __name__ == '__main__':
    tf.app.run()
