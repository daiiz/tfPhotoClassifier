# coding: utf8
# python gen_labeled_jsons.py --theme=sample
import os
import json
import glob
import random
import tensorflow as tf

tf.app.flags.DEFINE_string('theme', 'sample', 'theme')
FLAGS = tf.app.flags.FLAGS
tool_version = '0.0.1'

theme = FLAGS.theme


def make_labled_json (labels_set):
    num_classes = labels_set['num_classes']
    raw_json_dir = 'workspace/{}/raw_jsons'.format(theme)
    raw_jsons = glob.glob('{}/*.json'.format(raw_json_dir))

    # labeled_jsonのデータ数を管理する
    # train, evalの添字はラベル番号に対応している
    label_nums = {
        'train': [],
        'eval': []
    }
    for j in range(num_classes):
        label_nums['train'].append(0)
        label_nums['eval'].append(0)

    # ラベル番号毎に繰り返し処理をする
    for idx in range(num_classes):
        labels = labels_set['labels'][idx]
        keep_items = []
        # ディレクトリ内のJSONファイルをすべて読む
        for json_file in raw_jsons:
            f = open(json_file)
            photo_data = json.load(f)
            f.close()
            items = photo_data['items']
            # JSONファイル内のアイテムを1つずつ読んで，ラベルが一致したものをキープする
            for item in items:
                item_labels = item['labels']
                # itemに付与されたラベルを1つずつ読む
                for lb in item_labels:
                    if lb in labels:
                        keep_items.append(item)
        # キープしたラベルアイテムの個数を表示して，これらのうちいくつを評価用にまわすかを尋ねる
        len_labeled_json = len(keep_items)
        print('Label: {} ({}), total_examples_nums={}, eval_nums?...'.format(idx, labels, len_labeled_json))
        len_eval_json = int(raw_input())  # ユーザから入力を受け取る

        if len_eval_json <= len_labeled_json:
            eval_labeled_items = []
            train_labeled_items = []
            # キープしたアイテムをシャッフルしてランダムに並び替える
            random.shuffle(keep_items)
            # 末尾から指定件数ぶんを評価用に取る
            for i in range(len_eval_json):
                eval_labeled_items.append(keep_items.pop())
            train_labeled_items = keep_items

            label_nums['train'][idx] = len(train_labeled_items)
            label_nums['eval'][idx] = len(eval_labeled_items)

            save_photocropper_json(train_labeled_items, 'photocropper-{}.json'.format(idx))
            save_photocropper_json(eval_labeled_items, 'eval-photocropper-{}.json'.format(idx))

    # cifar10.labels.jsonにlabel_numsを追記して保存する
    labels_file_path = 'workspace/{}/cifar10.labels.json'.format(theme)
    f = open(labels_file_path)
    labels_data = json.load(f)
    f.close()
    # 訓練用と評価用のデータ例の個数をラベルごとに記録する
    labels_data['num_examples'] = label_nums
    dump_json(labels_data, labels_file_path)

# photocropper-LABEL_NUM.jsonを出力する
def save_photocropper_json (items, file_name):
    file_path = 'workspace/{}/{}'.format(theme, file_name)
    res = {
        'items': items,
        'version': tool_version
    }
    print('Saved! {}'.format(file_path))
    dump_json(res, file_path)


def dump_json (save_dict, file_path):
    text = json.dumps(save_dict, ensure_ascii=False, indent=4)
    with open(file_path, 'w') as fh:
        fh.write(text.encode('utf-8'))

def load_labeles_file ():
    # 分類グループ数とラベルを取得する
    # labelsの添字がlabel_numberとなる
    # labelsの要素はtool-labelesリストである
    res = {
        'num_classes': None,
        'labels': [],
        'answer_expression': []
    }
    file_path = 'workspace/{}/cifar10.labels'.format(theme)
    for line in open(file_path, 'r'):
        line = line.strip()
        if line != '' and line[0] != '#':
            if res['num_classes'] == None:
                res['num_classes'] = int(line)
                n = res['num_classes']
                # 分類数ぶん，placeholderとしてlabel_numberを詰め込む
                for idx in range(n):
                    res['labels'].append([str(idx)])
                    res['answer_expression'].append(str(idx))
            else:
                tokens = line.split(',')
                if len(tokens) == 3:
                    idx = int(tokens[0].strip())
                    labels = map(lambda label: label.strip(), tokens[1].strip().split('|'))
                    ans = tokens[2].strip()
                    # labelを追加する
                    for label in labels:
                        if not label in res['labels'][idx]:
                            res['labels'][idx].append(label)
                    # answerを追加する
                    res['answer_expression'][idx] = ans
    return res

if __name__ == '__main__':
    labels_set = load_labeles_file()

    # 保存しておく
    dump_json(labels_set, 'workspace/{}/cifar10.labels.json'.format(theme))

    make_labled_json(labels_set)
