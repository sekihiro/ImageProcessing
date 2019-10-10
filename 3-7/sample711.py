#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import numpy as np

# ---------- 拡大 ----------
# 最近傍補間 (ニアレストネイバー)
# バイリニア補間
# バイキュービック補間

IMG_PATH = '/home/seki/sample_data/city.jpg'

def main():

    #  入力画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    # 画像サイズを2倍に拡大
    # 第1引数:画像
    # 第2引数:リサイズしたい(幅, 高さ)のタプル
    # 第3引数:拡大・縮小の補完方法 (下の補完方法になるにつれて計算量が増えるが画像がなめらかになる)
    # cv2.INTER_NEAREST 最近傍補間          [補間対象となるピクセルに最も近い画素位置の値をそのまま利用]
    # cv2.INTER_LINEAR バイリニア補間       [補間対象となるピクセルを、周辺の4個(2×2)のピクセルを参照した上で割り出す]
    # cv2.INTER_AREA 平均画素法(面積平均法) [ピクセルの面積比を考慮して平均して補間する]
    # cv2.INTER_CUBIC                       [補間対象となるピクセルを、周辺の16(4×4)個のピクセルを参照し、距離を考量して割り出す]
    # cv2.INTER_LANCZOS4                    [8×8 の近傍領域を利用する Lanczos法の補間]
    nearest = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation = cv2.INTER_NEAREST)
    linear = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation = cv2.INTER_LINEAR)
    area = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation = cv2.INTER_AREA)
    cubic = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation = cv2.INTER_CUBIC)
    lanczos = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation = cv2.INTER_LANCZOS4)

    #  結果を出力
    cv2.imwrite(bufs[0] + '_00_gray' + bufs[1], gray)
    cv2.imwrite(bufs[0] + '_01_nearest' + bufs[1], nearest)
    cv2.imwrite(bufs[0] + '_02_linear' + bufs[1], linear)
    cv2.imwrite(bufs[0] + '_03_area' + bufs[1], area)
    cv2.imwrite(bufs[0] + '_04_cubic' + bufs[1], cubic)
    cv2.imwrite(bufs[0] + '_05_lanczos' + bufs[1], lanczos)


if __name__ == "__main__":
    main()

# EOF #