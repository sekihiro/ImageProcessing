#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np

import matplotlib
matplotlib.use('Agg')   # savefig()用
import matplotlib.pyplot as plt

# ---------- 空間周波数フィルタリング ----------
# パワースペクトル

# フーリエ変換とは、ざっくりいうと複雑な変化から隠れた性質を抜き出す方法
# FFTを行うときは画素数は2の倍数でなければいけない
# パワースペクトルの中心部分が低周波成分、外側の方へ高周波成分となっています。画像を周波数の方向からみることによって、違った見方ができるようになる
# 複雑な変化の裏側を見るという点においてフーリエ変換は、絶大な強さを誇る

#IMG_PATH = '/home/seki/book/ImageProcessingAlgorithm/3-9/input.png'
IMG_PATH = '/home/seki/sample_data/lena.png'

def main():

    # 画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    # ***** FFTを行うときは画素数は2の倍数でなければいけない？ *****
    (h, w) = gray.shape
    print('height : {}, width : {}'.format(h, w))

    # 高速フーリエ変換(2次元)
    fimg = np.fft.fft2(gray)

    # 2次元FFTした結果は、直流成分が配列の左上にある
    # パワースペクトルを見やすくするために、np.fft.fftshift()を使って直流成分を配列の中心に移動させる
    # np.fft.fftshift()は、配列の第1象限と第4象限、第2象限と第3象限をそれぞれ入れ替える
    fimg =  np.fft.fftshift(fimg)

    # パワースペクトルの計算
    mag = 20 * np.log(np.abs(fimg))

    # 入力画像とスペクトル画像をグラフ描画
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap = 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mag, cmap = 'gray')

    #  結果を出力
    plt.savefig(bufs[0] + '_00_mag' + bufs[1])


if __name__ == "__main__":
    main()

# EOF #