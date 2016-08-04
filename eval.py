# coding: utf8
# python eval.py --theme=sample
import os
from models.image.cifar10 import cifar10
import tensorflow as tf
import numpy as np
import math
import datetime as dt
from common import *

FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 2

tf.app.flags.DEFINE_string('theme', 'theme', 'theme')
theme = FLAGS.theme
FLAGS.batch_size  = BATCH_SIZE

size = get_size()

cifar10.IMAGE_SIZE = size['width']
cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = get_num_examples_per_epoch_for_eval(theme)


def distorted_inputs (tfrecord_file_paths=[]):
    fqueue = tf.train.string_input_producer(tfrecord_file_paths)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(fqueue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    })
    image = tf.image.decode_jpeg(features['image'], channels=size['depth'])
    image = tf.cast(image, tf.float32)
    image.set_shape([size['width'], size['height'], size['depth']])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL * min_fraction_of_examples_in_queue)

    images, labels = tf.train.shuffle_batch(
        [tf.image.per_image_whitening(image), tf.cast(features['label'], tf.int32)],
        batch_size=BATCH_SIZE,
        capacity=min_queue_examples + 3 * BATCH_SIZE,
        min_after_dequeue=min_queue_examples
    )

    images = tf.image.resize_images(images, size['input_width'], size['input_height'])
    tf.image_summary('images', images)
    return images, labels


def eval_once (theme, saver, summary_writer, top_k_op, summary_op):
    checkpoint_path = 'workspace/{}/train'.format(theme)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / BATCH_SIZE))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
            # Compute precision @ 1.
            precision = 1.0 * true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (dt.datetime.now(), precision))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate (tfrecord_file_paths, theme):
    eval_dir = 'workspace/{}/eval'.format(theme)
    with tf.Graph().as_default() as g:
        images, labels = distorted_inputs(tfrecord_file_paths=tfrecord_file_paths)
        logits = cifar10.inference(tf.image.resize_images(images, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE))

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = {}

        for v in tf.all_variables():
            if v in tf.trainable_variables():
                restore_name = variable_averages.average_name(v)
            else:
                restore_name = v.op.name
            variables_to_restore[restore_name] = v

        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(eval_dir, g)

        eval_once(theme, saver, summary_writer, top_k_op, summary_op)


if __name__ == '__main__':
    cifar10.NUM_CLASSES = get_num_classes(theme)

    file_paths = []
    for i in range(cifar10.NUM_CLASSES):
        file_paths.append('workspace/{}/tfrecords/{}-eval-data{}.tfrecords'.format(theme, theme, i))

    evaluate(tfrecord_file_paths=file_paths, theme=theme)
