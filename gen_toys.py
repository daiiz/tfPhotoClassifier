# coding: utf8
# python gen_toys.py --theme=sample --ans=0
import os
import json
import base64
import tensorflow as tf
import random

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('theme', 'sample', 'theme')
tf.app.flags.DEFINE_string('ans', 'ans', 'answer')

def random_save (jsonfile, limit=5):
    theme = FLAGS.theme
    ans = FLAGS.ans
    f = open(jsonfile)
    photo_data = json.load(f)
    f.close()

    items = photo_data['items']
    if len(items) < limit:
        return
    random.shuffle(items)

    for i in range(limit):
        toy_jpg_path = 'workspace/{}/toys/toy-ans{}-{}.jpg'.format(theme, ans, i)
        item = items[i]
        base64code = item['img_base64_cropped']
        encoded = base64code.replace('data:image/jpeg;base64,', '')
        img = encoded.decode('base64')
        g = open(toy_jpg_path, 'wb')
        g.write(img)
        g.close()


if __name__ == '__main__':
    eval_json_path = 'workspace/{}/eval-photocropper-{}.json'.format(FLAGS.theme, FLAGS.ans)

    random_save(jsonfile=eval_json_path, limit=5)
