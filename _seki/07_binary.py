#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import os
import numpy as np

# ---------- 二値化 ----------
# 単純
# 適応的閾値処理
# 

IMG_PATH = '/home/seki/sample_data/animal.jpg'

def main():

    #  入力画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    # 単純二値化処理
    # 画素値が閾値より大きければある値(白色)を割り当て、そうでなければ別の値(黒色)を割り当てる
    # threshold : 閾値
    # maxValue : 条件を満足するピクセルに割り当てられる非0の値 (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV の場合に使用する)
    # thresholdType : cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV
    (ret, th) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 適応的閾値処理
    # 画像の小領域ごとに閾値の値を計算する。そのため領域によって光源環境が変わるような画像に対しては、単純な閾値処理より良い結果が得られる。
    # maxValue – 条件を満足するピクセルに割り当てられる非0の値。
    # adaptiveMethod – 利用される適応的閾値アルゴリズム： ADAPTIVE_THRESH_MEAN_C または ADAPTIVE_THRESH_GAUSSIAN_C
    # thresholdType – 閾値の種類． THRESH_BINARY または THRESH_BINARY_INV のどちらか
    # blockSize – ピクセルの閾値を求めるために利用される近傍領域のサイズ．3, 5, 7, など
    # C – 平均または加重平均から引かれる定数
    adp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 大津の二値化
    # 自動的に閾値を決定して二値化処理
    (ret_otsu, otsu) = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    print('ret_otsu : {}'.format(ret_otsu))

    #  結果を出力
    cv2.imwrite(bufs[0] + '_00_gray' + bufs[1], gray)
    cv2.imwrite(bufs[0] + '_10_threshold' + bufs[1], th)
    cv2.imwrite(bufs[0] + '_20_adaptive' + bufs[1], adp)
    cv2.imwrite(bufs[0] + '_30_otsu' + bufs[1], otsu)


if __name__ == "__main__":
    main()

# EOF #