#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# OpenCVの関数を使う

# メディアン
# ガウシアン
# ラプラシアン
# ソーベル
# Canny

IMG_PATH = '/home/seki/sample_data/dog.jpg'

def main():

    #  入力画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    # エッジを保持した平滑化
    # メディアンフィルタ処理
    # 領域内の中央値を出力
    # ksize     カーネルサイズ
    median3 = cv2.medianBlur(gray, ksize=3)
    median5 = cv2.medianBlur(gray, ksize=5)

    # 平滑化
    # ガウシアンフィルタ処理
    # フィルタ原点に近いほど重み付き平均
    # ksize     カーネルサイズ
    # sigmaX    ガウス分布のsigma_x
    gaussian3 = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=1.3)
    gaussian5 = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=1.3)

    # エッジ抽出
    # ソーベルフィルタ処理
    # 原画像を一次微分でエッジを検出する関数で、平滑化する効果もある。
    # プレウィットフィルタは「微分フィルタ」と「平滑化フィルタ」を組み合わせることで、ノイズの影響を抑えながら輪郭を抽出する。
    # この平滑化フィルタをかける際に「注目画素との距離に応じて重み付けを変化させた」ものがソーベルフィルタで、より自然に平滑化を行うことが出来る。
    # ddepth: 出力の色深度 (cv2.CV_32Fが推奨)
    # dx: x方向の微分の次数  (x,y)=(1,0) ⇒ X方向用
    # dy: y方向の微分の次数  (x,y)=(0,1) ⇒ Y方向用
    # ksize: カーネルサイズ、1, 3, 5, 7のどれかを指定
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # エッジ抽出
    # ラプラシアンフィルタ処理
    # 二次微分を用いて画像ピクセル値の変化を検出する関数で、僅かな変化にも敏感に反応し、ノイズに弱い。
    # ガウシアンフィルタなどでノイズを除去してから使用するのが原則である。Sobelのように方向性を持たない。
    # ddepth: 出力の色深度 (cv2.CV_32Fが推奨)
    # ksize: カーネルサイズ、1, 3, 5, 7のどれかを指定
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    # Cannyアルゴリズム
    #  ・ガウシアンフィルタで平滑化
    #  ・Sobelフィルタを使って縦方向と横方向の1次微分を取得し、2つの微分画像からエッジ勾配と方向を求める
    #  ・エッジと関係ない画素を取り除く (非極大値の抑制)
    #  ・Hysteresisを使ったしきい値処理
    # threshold1: ヒステリシス処理の最小閾値
    # threshold2: ヒステリシス処理の最大閾値
    # 空間フィルタ(ソーベルやラプラシアン等)を使ったエッジ抽出による輪郭線の検出は、エッジが不連続でノイズが多い傾向にある
    # Cannyは、未検出、誤検出が少なく、検出位置が正確である
    canny = cv2.Canny(gray, 100, 200)

    # 8ビット符号なし整数に変換
    #laplacian2 = cv2.convertScaleAbs(laplacian)

    #  結果を出力
    cv2.imwrite(bufs[0] + '_00_gray' + bufs[1], gray)
    cv2.imwrite(bufs[0] + '_01_median3' + bufs[1], median3)
    cv2.imwrite(bufs[0] + '_01_median5' + bufs[1], median5)
    cv2.imwrite(bufs[0] + '_02_gaussian3' + bufs[1], gaussian3)
    cv2.imwrite(bufs[0] + '_02_gaussian5' + bufs[1], gaussian5)
    cv2.imwrite(bufs[0] + '_11_sobelx' + bufs[1], sobelx)
    cv2.imwrite(bufs[0] + '_11_sobely' + bufs[1], sobely)
    cv2.imwrite(bufs[0] + '_12_laplacian' + bufs[1], laplacian)
    #cv2.imwrite(bufs[0] + '_laplacian2' + bufs[1], laplacian2)
    cv2.imwrite(bufs[0] + '_13_canny' + bufs[1], canny)


if __name__ == "__main__":
    main()