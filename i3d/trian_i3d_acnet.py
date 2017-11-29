"""Script for training I3D model on ActivityNet for classification."""

from __future__ import absolute_import
from __future__ import division
from datetime import datetime
import os.path
import time
import collections

import numpy as np
import tensorflow as tf

import i3d

from acnet_reader import get_rgb_input
import train_i3d_acnet_config as config

_IMAGE_SIZE = 224
_NUM_CLASSES = 51

base_dir = '/ssd2/hmdb/'

max_steps = 5000

_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
tf.flags.DEFINE_string("train_dir", "",
                       "Frequency at which validation loss is computed for one mini-batch.")
tf.flags.DEFINE_string("summary_dir", "",
                       "Frequency at which validation loss is computed for one mini-batch.")


def i3d_loss(rgb, label, rgb_model, gpu):
    """
    Builds an I3D model and computes the loss
    """
    rgb_logits = rgb_model(rgb, is_training=True, dropout_keep_prob=1.0)[0]
    rgb_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=rgb_logits, name='cross_entropy_rgb')
    ce_rgb = tf.reduce_mean(rgb_loss, name='rgb_ce')
    tf.summary.scalar('rgb_%d' % gpu, ce_rgb)
    return ce_rgb


def average_gradients(grads):
    """
    Averages all the gradients across the GPUs.
    """
    average_grads = []
    for grad_and_vars in zip(*grads):
        gr = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            exp_g = tf.expand_dims(g, 0)
            gr.append(exp_g)
        if len(gr) == 0:
            continue
        grad = tf.concat(axis=0, values=gr)
        grad = tf.reduce_mean(grad, 0)

        # remove redundant vars (because they are shared across all GPUs)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(batch_size=config.batch_size, num_gpus=3):
    assert FLAGS.train_dir, "--train_dir is required"
    assert FLAGS.summary_dir, "--summary_dir is required"

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create the network.
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')

        # Initial learning rate.
        lr = tf.Variable(config.initial_learning_rate)

        # Create optimizer.
        opt = tf.train.MomentumOptimizer(lr, config.momentum)

        # Get data set.
        rgbs, labels = get_rgb_input(base_dir, os.path.join(base_dir, 'splits/final_split1_train.txt'), batch_size)
        batch_queue_capacity = 2 * config.batch_size
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([rgbs, labels], capacity=batch_queue_capacity)

        # Create model on each GPU and get gradients for each
        all_rgb_grads = []
        rgb_loss = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(num_gpus):
                with tf.device('gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('I3D', i)) as scope:
                        # Dequeue a batch.
                        rgb_batch, label_batch = batch_queue.dequeue()
                        rgb_loss, flow_loss = i3d_loss(scope, rgb_batch, None, label_batch, rgb_model, None, i)

                        # Reuse the across models on different GPUs.
                        tf.get_variable_scope().reuse_variables()

                        # calculate gradients for this tower
                        # track all gradients
                        if rgb_loss is not None:
                            grads_rgb = opt.compute_gradients(rgb_loss)
                            all_rgb_grads.append(grads_rgb)

                        # Retain summaries.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        # sync and average grads
        rgb_grads = average_gradients(all_rgb_grads)

        # track lr
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # track grads:
        for grad, var in rgb_grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/rgb_gradients', grad))

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        # Initialize new layers (layers that are not defined in the pre-trained model).
        new_layers = [lr]

        # Load pre-trained weights.
        rgb_variable_map = {}
        rgb_final_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                if 'Logits' in variable.name:
                    new_layers.append(variable)
                else:
                    rgb_variable_map[variable.name.replace(':0', '')] = variable
                rgb_final_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        rgb_saver.restore(sess, config.pretrained_model_paths['rgb_imagenet'])
        rgb_saver = tf.train.Saver(var_list=rgb_final_map, reshape=True)

        # Apply gradients.
        apply_rgb_grad_op = opt.apply_gradients(rgb_grads)  # , global_step=global_step)

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        train_op = apply_rgb_grad_op

        summary_op = tf.summary.merge(summaries)

        # init all momentum vars
        for var in tf.global_variables():
            if 'Momentum' in var.name:
                new_layers.append(var)
        # init new layers
        init = tf.variables_initializer(new_layers)
        sess.run(init)

        # Start queue runners.
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
        loss = rgb_loss

        # Begin Training Iterations.
        losses = collections.deque(maxlen=10)
        last_loss = None
        for step in xrange(max_steps):
            start_time = time.time()
            _, loss_value, lra = sess.run([train_op, loss, lr])
            duration = time.time() - start_time

            # Exponentially decay learning rate when we find loss saturates.
            if step % 10 == 0 and step > 0:
                # check if loss is saturated
                if last_loss is None:
                    last_loss = sum(losses) / 10
                else:
                    diff = last_loss - sum(losses) / 10
                    print 'Diff:', diff
                    last_loss = sum(losses) / 10
                    if diff < 0.001:
                        lr /= 10.

            num_examples_per_step = batch_size * num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / num_gpus
            format_str = ('%s: step %d, avg loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), step, sum(losses) / 10,
                                 examples_per_sec, sec_per_batch))
            print 'learning rate: %f' % lra

            if step % config.save_summary_every_n_iterations == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % config.save_model_every_n_iterations == 0 or (step + 1) == max_steps:
                model_save_path = os.path.join(FLAGS.train_dir, config.pretrained_model_paths)
                rgb_saver.save(sess, model_save_path, global_step=step)


if __name__ == '__main__':
    tf.app.run(train)
