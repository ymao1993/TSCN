from __future__ import unicode_literals, division

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

kNumFramesPerSnippets = 64
kI3DResizedImageWidth = 256
kI3DResizedImageHeight = 256
kI3DFinalImageWidth = 224
kI3DFinalImageHeight = 224
kI3DImageChannels = 3


def fix_image_flip_shape(image_shape, result):
    """Set the shape to 3 dimensional if we don't know anything else.
    Args:
      image: original image size
      result: flipped or transformed image
    Returns:
      An image whose shape is at least None,None,None.
    """
    from tensorflow.python.framework import tensor_shape

    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None, None])
    else:
        result.set_shape(image_shape)
    return result


def random_flip_left_right(image, seed=None):
    """Randomly flip an image horizontally (left to right).
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    Args:
      image: A 3-D tensor of shape `[frames, height, width, channels].`
      seed: A Python integer. Used to create a random seed. See
        @{tf.set_random_seed}
        for behavior.
    Returns:
      A 3-D tensor of the same type and shape as `image`.
    Raises:
      ValueError: if the shape of `image` not supported.
    """
    image = ops.convert_to_tensor(image, name='image')
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = math_ops.less(uniform_random, .5)
    result = control_flow_ops.cond(mirror_cond,
                                   lambda: array_ops.reverse(image, [2]),
                                   lambda: image)
    return fix_image_flip_shape(image.get_shape(), result)


def read_image_frame(base, video_dir, start, num):
    """
    Read RGB images from [base/video_dir/start.jpg, base/video_dir/start+num-1.jpg]
    Note that at least one of base, video_dir, start should be a tensor. (Because this is part of the graph!)
    :return:
    """
    imgs = []
    for i in range(num):
        # convert to file name in tf format
        path = tf.string_join(
            [tf.constant(base) + '/', video_dir + '/', tf.as_string(i + start, width=6, fill='0'),
             tf.constant('.jpg')])
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        imgs.append(img)
    return tf.stack(imgs)


def preprocess_frames(frames):
    """ Pre-process RGB image frame by cropping, randomly flipping, re-scaling.
    """
    cropped_shape = (kNumFramesPerSnippets, kI3DFinalImageWidth, kI3DFinalImageHeight, kI3DImageChannels)
    # take random crops
    rgb_frames = tf.random_crop(frames, size=cropped_shape)
    # random left/right flip
    rgb_frames = random_flip_left_right(rgb_frames)
    # Scale the image pixel values from [0, 1] to [-1, 1]
    rgb_frames *= 2.
    rgb_frames -= 1.
    rgb_frames.set_shape(cropped_shape)
    return rgb_frames


def read_video_frames_and_preprocess(queue, base):
    """ Read video snippet randomly from the video folder in the queue.
    """
    csv_line = queue.dequeue()
    video_name, num_frames, label = tf.decode_csv(csv_line, [[''], [0], [-1]], field_delim=',')
    start = tf.random_uniform((), 1, num_frames - kNumFramesPerSnippets + 2, dtype=tf.int32)
    frames = read_image_frame(base, video_name, start, kNumFramesPerSnippets)
    frames_preprocessed = preprocess_frames(frames)
    label = tf.cast(label, tf.int32)
    return frames_preprocessed, label


def generate_batch(frame, label, batch_size):
    """ Batch frame stream and provide mini-batch of frame, label pairs to the downstream component.
    """
    # Number of threads used to enqueue [frame, label],
    # which equals to the number of threads to perform
    # read_video_frames_and_preprocess() in this context.
    num_preprocess_threads = 16
    stream, labels = tf.train.shuffle_batch([frame, label],
                                            batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=256,
                                            min_after_dequeue=32)
    return stream, tf.reshape(labels, [batch_size])


def get_rgb_input(data_dir, split_file, batch_size):
    """Returns images, and labels of size [batch, 64, 224, 224, 3]
    """

    with open(split_file, 'r') as f:
        files = f.read().splitlines()

    # create a queue to select videos to read
    file_queue = tf.train.string_input_producer(files)

    # read the video
    frame, label = read_video_frames_and_preprocess(file_queue, data_dir)

    return generate_batch(frame, label, batch_size)
