# -*- coding: utf8 -*-
# python play.py --theme=sample --jpg=workspace/sample/toys/toy-ans0-0.jpg
# python play.py --theme=sample --toyjpg=toy-ans0-0

import os
from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
from common import *

tf.app.flags.DEFINE_string('theme', 'sample', 'theme')
tf.app.flags.DEFINE_string('jpg', '', '.jpg file')
tf.app.flags.DEFINE_string('toyjpg', '', 'workspace/theme/toys/*.jpg file')
FLAGS = tf.app.flags.FLAGS
FLAGS.batch_size = 1

def detect_input_file ():
    input_image_file = ''
    # 入力画像ファイルパスを決定する
    if FLAGS.toyjpg != '':
        input_image_file = 'workspace/{}/toys/{}.jpg'.format(FLAGS.theme, FLAGS.toyjpg)
    if FLAGS.jpg != '':
        input_image_file = FLAGS.jpg
    if input_image_file == '':
        return None
    return input_image_file


def play_main (img):
    size = get_size()
    # 学習結果を利用するための準備
    cifar10.IMAGE_SIZE = size['width']
    cifar10.NUM_CLASSES = get_num_classes(FLAGS.theme)
    checkpoint_path = 'workspace/{}/train'.format(FLAGS.theme)
    # 入力は画像1枚分なので，FRAGS.batch_size=1としておく
    images = tf.placeholder(tf.float32, shape=(1, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, size['depth']))
    logits = tf.nn.softmax(cifar10.inference(images))

    sess = tf.Session()
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No ckpt file')
    # 識別器での判定処理
    decoded = tf.image.decode_jpeg(img, channels=3)
    inputs = tf.reshape(decoded, decoded.eval(session=tf.Session()).shape)
    inputs = tf.image.per_image_whitening(inputs)
    inputs = tf.image.resize_images(tf.expand_dims(inputs, 0), cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE)
    output = sess.run(logits, feed_dict={images: inputs.eval(session=tf.Session())}).flatten().tolist()
    return output


if __name__ == '__main__':
    input_img_file = detect_input_file()
    # 画像ファイルを読み込む
    with open(input_img_file) as f:
        output = play_main(img=f.read())
        print_answer(FLAGS.theme, output)
