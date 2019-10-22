#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# ヒストグラム平均化 #

IMG_PATH = '/home/seki/sample_data/dog.jpg'

def main():
    #  入力画像を読み込み
    img = cv2.imread(IMG_PATH)

    #  ヒストグラム算出
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #  グラフの作成
    plt.xlim(0, 255)
    plt.plot(hist)
    plt.xlabel("Pixel value", fontsize=20)
    plt.ylabel("Number of pixels", fontsize=20)
    plt.grid()
    plt.show()

    #  入力画像をグレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #   ヒストグラム平均化
    dst2 = cv2.equalizeHist(gray)

    #  結果の出力
    dir_name = os.path.dirname(IMG_PATH)
    base_name = os.path.basename(IMG_PATH)
    bufs = os.path.splitext(IMG_PATH)
    cv2.imwrite(bufs[0] + '_equalizeHist' + bufs[1], dst2)
    cv2.imshow("equalizeHist", dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #  ヒストグラム算出
    hist = cv2.calcHist([dst2], [0], None, [256], [0, 256])
    #  グラフの作成
    plt.xlim(0, 255)
    plt.plot(hist)
    plt.xlabel("Pixel value", fontsize=20)
    plt.ylabel("Number of pixels", fontsize=20)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()