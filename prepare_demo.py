# coding: utf-8
import os
import argparse
import glob
import sys

def gen_photolist_js (theme):
    print(theme)
    dir_theme_workspace = os.path.abspath('./workspace/%s' % theme)
    photocropper_jsons = glob.glob('%s/photocropper-*.json' % dir_theme_workspace)
    ag = sys.argv
    
    # 写真情報
    f = open('./demo/photocropper.js', 'w')
    f.write('// $ python %s\n' % ' '.join(map(str, ag)))
    f.write('var photocropper = {};\n\n')
    f.write('photocropper.theme = "%s";\n' % theme)
    for label_num, photocropper_json in enumerate(photocropper_jsons):
        f.write('photocropper.label_%d = ' % label_num)
        
        photo_f = open(photocropper_json)
        # TODO: 枚数上限を設ける
        f.write(photo_f.read())
        f.write(';\n')
        photo_f.close()
    
    # ラベル情報
    label_f = open('%s/cifar10.labels.json' % dir_theme_workspace)
    f.write('photocropper.labels = ')
    f.write(label_f.read())
    f.write(';\n')
    label_f.close()
    
    f.close()

if __name__ == '__main__':
    argp =  argparse.ArgumentParser()
    argp.add_argument('-t', '--theme', default='sample')
    args = argp.parse_args()
    
    # workspace/THEME/photocropper-*.json を1つのJSファイルにまとめる
    gen_photolist_js(args.theme)
    