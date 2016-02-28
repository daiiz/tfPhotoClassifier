# coding: utf8
import os
import json

size = {
    # feed-cropped-image size
    # 32 x 32
    'width': 32,
    'height': 32,
    # inside 24 x 24 in 32 x 32
    'input_width': 24,
    'input_height': 24,
    # RGB
    'depth': 3,
    # ラベルの桁数
    'label': 1
}


def load_labeles_data (theme):
    labels_file_path = 'workspace/{}/cifar10.labels.json'.format(theme)
    f = open(labels_file_path)
    labels_data = json.load(f)
    f.close()
    return labels_data


# 分類数を返す
def get_num_classes (theme):
    labels_data = load_labeles_data(theme)
    return int(labels_data['num_classes'])


# 評価用の画像データの総数を返す
def get_num_examples_per_epoch_for_eval (theme):
    labels_data = load_labeles_data(theme)
    len_eval_examples = sum(labels_data['num_examples']['eval'])
    return len_eval_examples


# 訓練用の画像のデータ総数を返す
def get_num_examples_per_epoch_for_train (theme):
    labels_data = load_labeles_data(theme)
    len_train_examples = sum(labels_data['num_examples']['train'])
    return len_train_examples


# 最もスコアが高い項目を返す
def get_ans (theme, ans_list):
    labels_data = load_labeles_data(theme)
    answer_expressions = labels_data['answer_expression']
    return answer_expressions[ans_list.index(max(ans_list))]


def get_pretty_scores (theme, ans_list):
    labels_data = load_labeles_data(theme)
    answer_expressions = labels_data['answer_expression']
    num_classes = len(ans_list)
    # 一覧用のJSONを作る
    res = {}
    for i in range(num_classes):
        ans_expr = answer_expressions[i]
        res[str(ans_expr)] = ans_list[i]
    return res

# 答えを整形して表示する
def print_answer (theme, ans_list):
    res = get_pretty_scores(theme, ans_list)
    print(json.dumps(res, indent=4, ensure_ascii=False))
    # 最も数値が大きい答えを示す
    print(get_ans(theme, ans_list))


def get_size ():
    return size
