# -*- coding: utf8 -*-
# python gen_tfrecords.py --theme=sample

import os
import json
import base64
import tensorflow as tf
from common import *

from PIL import Image
import urllib

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('theme', 'sample', 'JSON dir, exported by tfPhotoPalette')
theme = FLAGS.theme

size = get_size()

def make_dataset (theme):
    num_classes = get_num_classes(theme)
    dataset = []
    # 訓練例用
    for i in range(num_classes):
        data = {
            'label': i,
            'json': 'photocropper-{}.json'.format(i),
            'name': 'data{}.tfrecords'.format(i)
        }
        dataset.append(data)

    # 評価用
    for i in range(num_classes):
        data = {
            'label': i,
            'json': 'eval-photocropper-{}.json'.format(i),
            'name': 'eval-data{}.tfrecords'.format(i)
        }
        dataset.append(data)

    return dataset


def load (photo_cropper_dir):
    dataset = make_dataset(photo_cropper_dir)
    for data in dataset:
        photo_cropper_file = 'workspace/' + photo_cropper_dir + '/' + data['json']
        label = data['label']
        f = open(photo_cropper_file)
        photo_data = json.load(f)
        f.close()

        items = photo_data['items']

        c = 0
        tfrecord_name = data['name']
        filename = os.path.join('workspace/{}/tfrecords'.format(photo_cropper_dir), photo_cropper_dir + '-' + tfrecord_name)
        writer = tf.python_io.TFRecordWriter(filename)

        for item in items:
            c += 1
            # Cropされた画像のbase64コードを取得
            base64code = item['img_base64_cropped']
            # デコードして画像を取得
            encoded = base64code.replace('data:image/jpeg;base64,', '')
            img = encoded.decode('base64')     # == urllib.urlopen(base64code).read()
            encode_cifar10(label=label, image=img, writer=writer)

        print('label: {}, len={}'.format(tfrecord_name, c))


def encode_cifar10 (label, image, writer):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
    }))
    writer.write(example.SerializeToString())


if __name__ == '__main__':
    dir_name = FLAGS.theme
    load(photo_cropper_dir=dir_name)
