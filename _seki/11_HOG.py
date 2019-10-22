#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np

# ---------- 人検出 ----------
# HoG 特徴量  + SVM
# HOG(Histograms of Oriented Gradients)とは局所領域の輝度の勾配方向をヒストグラム化したも”
# メリットとしては、勾配情報をもとにしているため、異なるサイズの画像を対象とする際も同じサイズにリサイズすることで比較可能

IMG_PATH = '/home/seki/sample_data/people.jpg'

def main():

    #  画像を読み込み
    img = cv2.imread(IMG_PATH)

    #  画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    # HoG 特徴量  + SVM で人の識別器を作成
    hog = cv2.HOGDescriptor()
    # cv2.HOGDescriptor_getDefaultPeopleDetector()      INRIA Person Dataset(64×128画素)のデータセットで学習している検出器
    # cv2.HOGDescriptor_getDaimlerPeopleDetector()      Daimler Pedestrian Detection Benchmark Dataset(48×96画素)のデータセットで学習している検出器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # 作成した識別器で人を検出
    # winStride ウィンドウの移動量
    # padding   入力画像の周囲の拡張範囲
    hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
    # 引数に*と**をつけると、任意の数の引数を指定することができる
    # *args: 複数の引数をタプルとして受け取る
    # **kwargs: 複数のキーワード引数を辞書として受け取る
    (human, r) = hog.detectMultiScale(gray, **hogParams)

    #  人の領域を赤色の矩形で囲む
    for (x, y, w, h) in human:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), 3)

    #  結果を出力
    cv2.imwrite(bufs[0] + '_00_HOG_detect' + bufs[1], img)

if __name__ == '__main__':
    main()

# EOF #