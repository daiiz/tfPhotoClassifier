# tfPhotoClassifier

## 新規テーマ作成
```
$ cd tfPhotoClassifier/
$ cp -r workspace/seed workspace/sample
```
上記を実行してsampleテーマを作成した場合，以降の手順中の`THEME`は「sample」を指す．

## 訓練画像収集，教師ラベル付け 〜 訓練 〜 評価
[tfPhotoPalette](http://daiiz.hatenablog.com/entry/2016/02/19/235524)でエクスポートしたJSONファイルを workspace/`THEME`/raw_jsons/ フォルダに保存する．<br>
cifar10.labelsで与えた内容に従って，tfPhotoPaletteで出力されたデータをラベルごとのJSONファイルに自動で分けられる．<br>
この際に収集した写真データを，訓練用と評価用に分けることができる．<br>
実行の途中で，各クラス（ラベル）に属する写真の総枚数が提示されて，そのうち何枚を評価用に割り振るかを指定できる．
```
$ python gen_labeled_jsons.py --theme=sample
```
最終的な生成物は workspace/`THEME`/ 直下に以下のような名前で配置される

* photocropper-`LABEL_NUMBER`.json
* eval-photocropper-`LABEL_NUMBER`.json

TensorFlow (Python version 2.7.10, TensorFlow version 0.6.0):
```
$ python gen_tfrecords.py --theme=sample
$ python train.py --theme=sample
$ python eval.py --theme=sample
```
TensorBoard:
```
$  tensorboard --logdir workspace/sample/train
$  tensorboard --logdir workspace/sample/eval
```

## 試して遊ぶ
#### CUI
規定サイズの任意のJPEG画像ファイルを渡して判定
```
$ python play.py --theme=sample --jpg=workspace/sample/toys/toy-ans0-0.jpg
$ python play.py --theme=sample --toyjpg=toy-ans0-0
```

評価用のデータセットの一部をJPEG画像ファイルとして出力
```
$ python gen_toys.py --theme=sample --ans=0
```

#### GUI
または，[tfPhotoPalette](https://chrome.google.com/webstore/detail/tfphotopalette/gcpfanfkkjpolcdicokfjphmdnelhbbb)を使用して判定
```
$ python palette_server.py --theme=sample
```
![https://gyazo.com/14f57b3eee0b2f636251a5e9f089984d](https://i.gyazo.com/14f57b3eee0b2f636251a5e9f089984d.png)
