#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import numpy as np

# ---------- 膨張縮小 ----------
# 膨張
# 縮小
# オープニング
# クロージング

IMG_PATH = '/home/seki/sample_data/animal.jpg'

def main():

    #  入力画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    #  二値化処理
    gray[gray<127] = 0
    gray[gray>=127] = 255

    # 8 近傍で処理
#    kernel = np.array([[1, 1, 1],
#                       [1, 1, 1],
#                       [1, 1, 1]], np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    # 膨張処理
    # src: 入力画像。
    # kernel: 構成要素。
    # iterations: 処理を何回繰り返すか。デフォルトは 1。
    dilate1 = cv2.dilate(gray, kernel, iterations = 1)
    dilate2 = cv2.dilate(gray, kernel, iterations = 2)

    # 収縮処理
    # src: 入力画像。
    # kernel: 構成要素。
    # iterations: 処理を何回繰り返すか。デフォルトは 1。
    erode1 = cv2.erode(gray, kernel, iterations = 1)
    erode2 = cv2.erode(gray, kernel, iterations = 2)

    # オープニング
    # 収縮の後に膨張をする処理でノイズ除去に有効
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations = 2)

    # クロージング
    # 膨張の後に収縮をする処理で小さな穴を埋めるのに有効
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # モルフォロジー勾配
    # 膨張した画像と収縮した画像の差分をとる処理
    # 結果として物体の外郭(境界線)が得られる
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations = 1)

    #  結果を出力
    cv2.imwrite(bufs[0] + '_00_gray' + bufs[1], gray)
    cv2.imwrite(bufs[0] + '_01_dilate1' + bufs[1], dilate1)
    cv2.imwrite(bufs[0] + '_01_dilate2' + bufs[1], dilate2)
    cv2.imwrite(bufs[0] + '_02_erode1' + bufs[1], erode1)
    cv2.imwrite(bufs[0] + '_02_erode2' + bufs[1], erode2)
    cv2.imwrite(bufs[0] + '_03_opening' + bufs[1], opening)
    cv2.imwrite(bufs[0] + '_04_closing' + bufs[1], closing)
    cv2.imwrite(bufs[0] + '_05_gradient' + bufs[1], gradient)

if __name__ == "__main__":
    main()

# EOF #