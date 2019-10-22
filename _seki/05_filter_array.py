#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# フィルタ配列を使う

# 平均化
# ガウシアン
# 微分フィルタ
# プリューウィット
# ソーベル
# ラプラシアン
# 鮮鋭化

IMG_PATH = '/home/seki/sample_data/dog.jpg'

def main():

    #  入力画像をグレースケールで読み込み
    print('loading image ...')
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    print('loading kernels ...')


    # ---------- 平滑化 ----------

    # 平均化
    kernel_average3 = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
     ])
    kernel_average5 = np.array([
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25]
     ])

    # ガウシアンフィルタ処理
    # フィルタ原点に近いほど大きな重みを付ける(加重平均)
    # 重みはガウス分布に従う
    kernel_gaussian3 = np.array([
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]
     ])
    kernel_gaussian5 = np.array([
        [1/256, 4/256, 6/256, 4/256, 1/256],
        [4/256, 16/256, 24/256, 16/256, 4/256],
        [6/256, 24/256, 36/256, 24/256, 6/256],
        [4/256, 16/256, 24/256, 16/256, 4/256],
        [1/256, 4/256, 6/256, 4/256, 1/256]
     ])


    # ---------- エッジ抽出 ----------

    # 微分フィルタ
    # 左右の差分値の平均をとって、注目画素の微分値とする。
    # 画像の濃淡が急激に変化するエッジ部分を抽出できるが、同時に画像に含まれるノイズを強調する傾向にある。
    kernel_differentialx = np.array([
        [0, 0, 0],
        [-1/2, 0, 1/2],
        [0, 0, 0]
     ])
    kernel_differentialy = np.array([
        [0, 1/2, 0],
        [0, 0, 0],
        [0, -1/2, 0]
     ])

    # プリューウィットフィルタ
    # 原画像を一次微分でエッジを検出する関数で、平滑化する効果もある。
    # 「微分フィルタ」と「平滑化フィルタ」を組み合わせることで、ノイズの影響を抑えながら輪郭を抽出する。
    kernel_prewittx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
     ])
    kernel_prewitty = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
     ])

    # ソーベルフィルタ
    # 原画像を一次微分でエッジを検出する関数で、平滑化する効果もある。
    # プリューウィットで平滑化フィルタをかける際に「注目画素との距離に応じて重み付けを変化させた」ものがソーベルフィルタで、より自然に平滑化を行うことが出来る。
    kernel_sobelx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
     ])
    kernel_sobely = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
     ])

    # ラプラシアンフィルタ処理
    # 二次微分を用いて画像ピクセル値の変化を検出する関数で、僅かな変化にも敏感に反応し、ノイズに弱い。
    # ガウシアンフィルタなどでノイズを除去してから使用するのが原則である。Sobelのように方向性を持たない。
    kernel_laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
     ])

    # ---------- 鮮鋭化 ----------
    # 元画像の濃淡を残したままエッジを強調する
    # 中心は 1+(8/9)k
    # 中心以外は -(k/9)
    # 3x3の値を全て足すと 1 となる
    # kの値が大きいほど鮮鋭化の度合いが増す
    kernel_sharpen1 = np.array([
        [-1/9, -1/9, -1/9],
        [-1/9, 17/9, -1/9],
        [-1/9, -1/9, -1/9]
     ])
    kernel_sharpen9 = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
     ])

    # フィルタリング
    # src: 入力画像
    # ddepth: 出力画像の型。デフォルトは -1。-1 の場合は入力画像と同じ型。
    # kernel: カーネル
    print('filtering ...')
    img_average3 = cv2.filter2D(gray, -1, kernel_average3)
    img_average5 = cv2.filter2D(gray, -1, kernel_average5)
    img_gaussian3 = cv2.filter2D(gray, -1, kernel_gaussian3)
    img_gaussian5 = cv2.filter2D(gray, -1, kernel_gaussian5)
    img_differentialx = cv2.filter2D(gray, -1, kernel_differentialx)
    img_differentialy = cv2.filter2D(gray, -1, kernel_differentialy)
    img_prewittx = cv2.filter2D(gray, -1, kernel_prewittx)
    img_prewitty = cv2.filter2D(gray, -1, kernel_prewitty)
    img_sobelx = cv2.filter2D(gray, -1, kernel_sobelx)
    img_sobely = cv2.filter2D(gray, -1, kernel_sobely)
    img_laplacian = cv2.filter2D(gray, -1, kernel_laplacian)
    img_sharpen1 = cv2.filter2D(gray, -1, kernel_sharpen1)
    img_sharpen9 = cv2.filter2D(gray, -1, kernel_sharpen9)

    #  結果を出力
    print('writing images ...')
    cv2.imwrite(bufs[0] + '_00_gray' + bufs[1], gray)
    cv2.imwrite(bufs[0] + '_01_average3' + bufs[1], img_average3)
    cv2.imwrite(bufs[0] + '_01_average5' + bufs[1], img_average5)
    cv2.imwrite(bufs[0] + '_02_gaussian3' + bufs[1], img_gaussian3)
    cv2.imwrite(bufs[0] + '_02_gaussian5' + bufs[1], img_gaussian5)
    cv2.imwrite(bufs[0] + '_11_differentialx' + bufs[1], img_differentialx)
    cv2.imwrite(bufs[0] + '_11_differentialy' + bufs[1], img_differentialy)
    cv2.imwrite(bufs[0] + '_12_prewittx' + bufs[1], img_prewittx)
    cv2.imwrite(bufs[0] + '_12_prewitty' + bufs[1], img_prewitty)
    cv2.imwrite(bufs[0] + '_13_sobelx' + bufs[1], img_sobelx)
    cv2.imwrite(bufs[0] + '_13_sobely' + bufs[1], img_sobely)
    cv2.imwrite(bufs[0] + '_14_laplacian' + bufs[1], img_laplacian)
    cv2.imwrite(bufs[0] + '_21_sharpen1' + bufs[1], img_sharpen1)
    cv2.imwrite(bufs[0] + '_21_sharpen9' + bufs[1], img_sharpen9)

    print('finish')


if __name__ == "__main__":
    main()

# EOF #
