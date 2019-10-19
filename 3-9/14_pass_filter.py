#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 空間周波数フィルタリング ----------
# ローパスフィルタ
# ハイパスフィルタ

#IMG_PATH = '/home/seki/book/ImageProcessingAlgorithm/3-9/input2.png'
IMG_PATH = '/home/seki/sample_data/input3.png'

def lowpass_ﬁlter(src, a = 0.5):

    # 高速フーリエ変換(2次元)
    fsrc = np.fft.fft2(src)

    # 画像サイズ
    (h, w) = fsrc.shape

    # 画像の中心座標
    cy, cx =  int(h/2), int(w/2)

    # フィルタのサイズ(矩形の高さと幅)
    (rh, rw) = int(a * cy), int(a * cx)

    # 2次元FFTした結果は、直流成分が配列の左上にある
    # パワースペクトルを見やすくするために、np.fft.fftshift()を使って直流成分を配列の中心に移動させる
    # np.fft.fftshift()は、配列の第1象限と第4象限、第2象限と第3象限をそれぞれ入れ替える
    fsrc =  np.fft.fftshift(fsrc)

    # 入力画像と同じサイズで値0の配列を生成
    # データ型は複素数(complex)
    fdst = np.zeros(src.shape, dtype = complex)

    # 中心部分の値だけ代入(中心部分以外は0のまま)
    # ローパスフィルタなので、中心部のみ通す
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = fsrc[cy-rh:cy+rh, cx-rw:cx+rw]

    # 配列の第1象限と第4象限、第2象限と第3象限をそれぞれ入れ替える(元に戻す)
    fdst =  np.fft.fftshift(fdst)

    # 高速逆フーリエ変換
    dst = np.fft.ifft2(fdst)

    # 実部の値のみを取り出し、符号なし整数型に変換して返す
    return  np.uint8(dst.real)

def highpass_filter(src, a = 0.5):

    # 高速フーリエ変換(2次元)
    fsrc = np.fft.fft2(src)

    # 画像サイズ
    (h, w) = fsrc.shape

    # 画像の中心座標
    cy, cx =  int(h/2), int(w/2)

    # フィルタのサイズ(矩形の高さと幅)
    (rh, rw) = int(a * cy), int(a * cx)

    # 2次元FFTした結果は、直流成分が配列の左上にある
    # パワースペクトルを見やすくするために、np.fft.fftshift()を使って直流成分を配列の中心に移動させる
    # np.fft.fftshift()は、配列の第1象限と第4象限、第2象限と第3象限をそれぞれ入れ替える
    fsrc =  np.fft.fftshift(fsrc)

    # 入力画像と同じ配列を生成
    fdst = fsrc.copy()

    # 中心部分だけ0を代入(中心部分以外は元のまま)
    # ハイパスフィルタなので、中心部だけ通さない
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0

    # 配列の第1象限と第4象限、第2象限と第3象限をそれぞれ入れ替える(元に戻す)
    fdst =  np.fft.fftshift(fdst)

    # 高速逆フーリエ変換
    dst = np.fft.ifft2(fdst)

    # 実部の値のみを取り出し、符号なし整数型に変換して返す
    return  np.uint8(dst.real)

def main():

    # 画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    #  ローパスフィルタ処理
    low_img = lowpass_ﬁlter(gray, 0.3)

    # ハイパスフィルタ処理
    high_img = highpass_filter(gray, 0.8)

    #  結果を出力
    cv2.imwrite(bufs[0] + '_00_gray' + bufs[1], gray)
    cv2.imwrite(bufs[0] + '_01_low-pass' + bufs[1], low_img)
    cv2.imwrite(bufs[0] + '_02_high-pass' + bufs[1], high_img)


if __name__ == "__main__":
    main()

# EOF #