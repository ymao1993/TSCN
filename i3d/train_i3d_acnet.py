"""Script for training I3D model on ActivityNet for classification."""

from __future__ import absolute_import
from __future__ import division
from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import i3d
from acnet_reader import get_rgb_input
import train_i3d_acnet_config as config

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("train_dir", "",
                       "Frequency at which validation loss is computed for one mini-batch.")
tf.flags.DEFINE_string("summary_dir", "",
                       "Frequency at which validation loss is computed for one mini-batch.")
tf.flags.DEFINE_integer("num_gpus", 1,
                        "Number of GPUs used for training.")
tf.flags.DEFINE_bool("freeze_up_to_logits", False,
                     "Freeze the layers up to logits layer (for fine-tuning).")


def i3d_loss(rgb, label, gpu_idx):
    """
    Builds an I3D model and computes the loss
    """
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(config.num_classes,
                                     spatial_squeeze=True,
                                     final_endpoint='Logits',
                                     freeze_before_logits=FLAGS.freeze_up_to_logits)
        rgb_logits = rgb_model(rgb, is_training=True, dropout_keep_prob=1.0)[0]
        rgb_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=rgb_logits, name='cross_entropy_rgb')
        loss_rgb = tf.reduce_mean(rgb_loss, name='rgb_ce')
        summary_loss = tf.summary.scalar('rgb_loss_%d' % gpu_idx, loss_rgb)
    return loss_rgb, summary_loss


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

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(args=None):
    assert FLAGS.train_dir, "--train_dir is required"
    assert FLAGS.summary_dir, "--summary_dir is required"
    assert FLAGS.num_gpus > 0, "--num_gpus must be greater or equals to 2"

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Initial learning rate.
        lr = tf.Variable(config.initial_learning_rate)

        # Create optimizer.
        opt = tf.train.MomentumOptimizer(lr, config.momentum)

        # Get training and validation data set.
        batch_queue_capacity = 2 * config.batch_size
        train_rgbs, train_labels = get_rgb_input(config.train_rgb_frames_base_dir,
                                                 config.train_video_meta_info_file,
                                                 config.batch_size)
        train_batch_queue = \
            tf.contrib.slim.prefetch_queue.prefetch_queue([train_rgbs, train_labels],
                                                          capacity=batch_queue_capacity)
        val_rgbs, val_labels = get_rgb_input(config.val_rgb_frames_base_dir,
                                             config.val_video_meta_info_file,
                                             config.batch_size)
        val_batch_queue = \
            tf.contrib.slim.prefetch_queue.prefetch_queue([val_rgbs, val_labels],
                                                          capacity=batch_queue_capacity)

        # Create model on each GPU and get gradients for each.
        all_rgb_grads = []
        train_rgb_losses = []
        val_rgb_loss = None
        train_summaries = []
        val_summaries = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('gpu:%d' % i):
                    # We use name scope so we can collect the gradients individually for each GPU.
                    with tf.name_scope('%s_%d' % ('Train_I3D', i)):
                        # De-queue a batch.
                        rgb_batch, label_batch = train_batch_queue.dequeue()
                        rgb_loss, summary_loss = i3d_loss(rgb_batch, label_batch, i)

                        train_rgb_losses.append(rgb_loss)

                        # Reuse the across models on different GPUs.
                        # This line of code works because the trainable variables are accessed
                        # using get_variable(). The name scoping mechanism ignores get_variable,
                        # but the variable scoping mechanism does not ignore get_variable. Therefore,
                        # because the variables across multiple GPUs share the same variable scope,
                        # the variables across multiple GPUs share the same name.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate gradients for this tower track all gradients.
                        grads_rgb = opt.compute_gradients(rgb_loss)
                        all_rgb_grads.append(grads_rgb)

                        # Collect the summary for the loss.
                        train_summaries.append(summary_loss)

                    # Use the last GPU for evaluation.
                    if i == FLAGS.num_gpus - 1:
                        # We use name scope so we can collect the gradients individually for each GPU.
                        with tf.name_scope('%s_%d' % ('Val_I3D', i)):
                            # De-queue a batch.
                            rgb_batch, label_batch = val_batch_queue.dequeue()
                            val_rgb_loss, _ = i3d_loss(rgb_batch, label_batch, i)

        # Sync and average grads.
        rgb_grads = average_gradients(all_rgb_grads)

        # Track learning rate.
        train_summaries.append(tf.summary.scalar('learning_rate', lr))

        # Track grads.
        for grad, var in rgb_grads:
            if grad is not None:
                train_summaries.append(tf.summary.histogram(var.op.name + '/rgb_gradients', grad))

        # Track average loss.
        avg_train_loss = tf.add_n(train_rgb_losses) / FLAGS.num_gpus
        train_summaries.append(tf.summary.scalar('Averaged Training Loss', avg_train_loss))

        # Track validation loss.
        val_summaries.append(tf.summary.scalar('Validation Loss', val_rgb_loss))

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
        rgb_saver.restore(sess, config.pretrained_model_paths[config.pretrained_model_used])
        rgb_saver = tf.train.Saver(var_list=rgb_final_map, reshape=True)

        # Apply gradients.
        apply_rgb_grad_op = opt.apply_gradients(rgb_grads)

        for var in tf.trainable_variables():
            train_summaries.append(tf.summary.histogram(var.op.name, var))

        train_op = apply_rgb_grad_op

        train_summary_op = tf.summary.merge(train_summaries)
        val_summary_op = tf.summary.merge(val_summaries)

        # init all momentum vars
        for var in tf.global_variables():
            if 'Momentum' in var.name:
                new_layers.append(var)
        # init new layers
        init = tf.variables_initializer(new_layers)
        sess.run(init)

        # Start queue runners.
        tf.train.start_queue_runners(sess=sess)
        train_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), sess.graph)
        val_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'val'), sess.graph)

        # Begin Training Iterations.
        for step in range(config.max_iteration):
            start_time = time.time()
            _, loss_value, lra, summary = sess.run([train_op, avg_train_loss, lr, train_summary_op])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            num_examples_per_step = config.batch_size * FLAGS.num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / FLAGS.num_gpus
            train_summary_writer.add_summary(summary, step)
            print ('%s: step %d, average loss = %f (%.1f examples/sec; %f sec/batch)' %
                   (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step % config.evaluate_model_every_n_iterations == 0:
                loss_value, summary = sess.run([val_rgb_loss, val_summary_op])
                print ('%s: step %d, validation loss = %f' %
                       (datetime.now(), step, loss_value))
                val_summary_writer.add_summary(summary, step)

            if step % config.save_model_every_n_iterations == 0 or (step + 1) == config.max_iteration:
                model_save_path = os.path.join(FLAGS.train_dir, config.model_save_name)
                rgb_saver.save(sess, model_save_path, global_step=step)


if __name__ == '__main__':
    tf.app.run(train)
