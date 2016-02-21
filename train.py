# coding: utf8
# python train.py --theme=sample
import os
import time
import numpy as np
from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
from common import *

from PIL import Image
import urllib

FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '128'))

tf.app.flags.DEFINE_string('theme', 'theme', 'theme')
theme = FLAGS.theme

size = get_size()

cifar10.IMAGE_SIZE = size['width']
cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = get_num_examples_per_epoch_for_train(theme)
cifar10.INITIAL_LEARNING_RATE = 0.09

def distorted_inputs (tfrecord_file_paths=[]):
    fqueue = tf.train.string_input_producer(tfrecord_file_paths)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(fqueue)
    features = tf.parse_single_example(serialized_example, dense_keys=['label', 'image'], dense_types=[tf.int64, tf.string])
    image = tf.image.decode_jpeg(features['image'], channels=size['depth'])
    image = tf.cast(image, tf.float32)
    image.set_shape([size['width'], size['height'], size['depth']])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    images, labels = tf.train.shuffle_batch(
        [tf.image.per_image_whitening(image), tf.cast(features['label'], tf.int32)],
        batch_size=BATCH_SIZE,
        capacity=min_queue_examples + 3 * BATCH_SIZE,
        min_after_dequeue=min_queue_examples
    )

    images = tf.image.resize_images(images, size['input_width'], size['input_height'])
    tf.image_summary('images', images)
    return images, labels


def train (tfrecord_file_paths, theme):
    train_dir = 'workspace/{}/train'.format(theme)
    max_steps = 10000
    global_step = tf.Variable(0, trainable=False)
    images, labels = distorted_inputs(tfrecord_file_paths=tfrecord_file_paths)
    logits = cifar10.inference(tf.image.resize_images(images, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE))
    loss = cifar10.loss(logits, labels)
    train_op = cifar10.train(loss, global_step)
    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        summary_writer = tf.train.SummaryWriter(train_dir)

        # 初期化
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.initialize_all_variables())

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        start = sess.run(global_step)

        for step in xrange(start, max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            print 'step=%d: loss = %f (%.3f sec/batch)' % (step, loss_value, duration)

            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 500 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

if __name__ == '__main__':
    cifar10.NUM_CLASSES = get_num_classes(theme)

    file_paths = []
    for i in range(cifar10.NUM_CLASSES):
        file_paths.append('workspace/{}/tfrecords/{}-data{}.tfrecords'.format(theme, theme, i))

    train(tfrecord_file_paths=file_paths, theme=theme)
