#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np

# ---------- 物体検出 ----------
# カスケード型識別器
# https://github.com/opencv/opencv/tree/master/data/haarcascades


IMG_PATH = '/home/seki/sample_data/faces.jpg'
CASCADE_FACE_XML = '/home/seki/book/ImageProcessingAlgorithm/haarcascades/haarcascade_frontalface_default.xml'
#CASCADE_FACE_XML = '/home/seki/book/ImageProcessingAlgorithm/haarcascades_cuda/haarcascade_frontalface_default.xml'

def main():

    #  画像を読み込み
    img = cv2.imread(IMG_PATH)

    #  画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(CASCADE_FACE_XML)

    # 検出実行
    # scaleFactor – 各画像スケールにおける縮小量を表します
    # minNeighbors – 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
    # minSize – 物体が取り得る最小サイズ．これよりも小さい物体は無視されます
    face = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # 領域を矩形で囲む
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 3)

    #  結果を出力
    cv2.imwrite(bufs[0] + '_00_cascade_detect' + bufs[1], img)


if __name__ == '__main__':
    main()

# EOF #