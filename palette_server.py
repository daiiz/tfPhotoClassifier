# coding: utf-8
# python palette_server.py --theme=sample --port=8000
from flask import Flask, request, jsonify
import urllib
from common import *
from models.image.cifar10 import cifar10
import tensorflow as tf

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello ():
    return 'Hello! This is palette_server.'


@app.route('/api/classify', methods=['POST'])
def classify ():
    theme = FLAGS.theme
    got_json = request.json
    if ('jpg' in got_json):
        base64img = got_json['jpg']
        img = urllib.urlopen(base64img).read()

        # 識別器での判定処理
        decoded = tf.image.decode_jpeg(img, channels=3)
        inputs = tf.reshape(decoded, decoded.eval(session=tf.Session()).shape)
        inputs = tf.image.per_image_whitening(inputs)
        inputs = tf.image.resize_images(tf.expand_dims(inputs, 0), cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE)
        scores = sess.run(logits, feed_dict={images: inputs.eval(session=tf.Session())}).flatten().tolist()

        ans = get_ans(theme=theme, ans_list=scores)
        scores = get_pretty_scores(theme=theme, ans_list=scores)
        return jsonify(theme=theme, scores=scores, description=ans)
    else:
        return jsonify()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('theme', 'sample', 'theme')
    tf.app.flags.DEFINE_string('port', '52892', 'port')
    FLAGS = tf.app.flags.FLAGS
    FLAGS.batch_size = 1

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

    app.run(port=FLAGS.port)
