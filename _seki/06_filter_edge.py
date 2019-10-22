#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# ---------- エッジを保持した平滑化 ----------
# メディアン
# バイラテラル
# ノンローカルミーン

IMG_PATH = '/home/seki/sample_data/dog.jpg'

def main():

    #  入力画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    # メディアンフィルタ処理
    # 領域内の中央値を出力
    # 入力画像のエッジがあまり影響を受けずに、スパイクノイズが除去できる
    # ksize     カーネルサイズ
    img_median3 = cv2.medianBlur(gray, ksize=3)
    img_median5 = cv2.medianBlur(gray, ksize=5)

    # バイラテラルフィルタ
    # 画素間のカラーとスペースの両者の距離に応じて近ければ近いほど重みが大きくなるフィルタ
    # 注目画素からの距離による重みに加えて、注目画素との画素値との差に応じて、ガウス分布に従う重みを付けた平滑化を行う
    # src: 入力画像
    # d: 注目画素をぼかすために使われる領域
    # sigmaColor: 色についての標準偏差。これが大きいと、画素値の差が大きくても大きな重み(強み平滑化)が採用される。
    # sigmaSpace: 距離についての標準偏差。これが大きいと、画素間の距離が広くても大きな重み(強み平滑化)が採用される。
    img_bilateral1 = cv2.bilateralFilter(gray, 5, 20, 20)
    img_bilateral2 = cv2.bilateralFilter(gray, 5, 20, 0.1)
    img_bilateral3 = cv2.bilateralFilter(gray, 5, 20, 10000)
    img_bilateral4 = cv2.bilateralFilter(gray, 5, 0.1, 20)
    img_bilateral5 = cv2.bilateralFilter(gray, 5, 100, 20)

    # ノンローカルミーンフィルタ
    # テンプレートマッチングのように周辺画素を含めた領域が、注目画素の周辺領域とどれくらい似通っているかによって重みを決定する
    # 元画像がgrayスケールで読み込むとエラーとなる
    # dst: None
    # h: 輝度成分のフィルタの平滑化の度合い、大きいとノイズが減少するが、エッジ部にも影響する
    # hColor: 色成分のフィルタの平滑化の度合い、10にしておけば十分
    # templateWindowSize: 周辺領域のテンプレートサイズ
    # searchWindowSize: 重みを探索する領域サイズ
    img_nlm1 = cv2.fastNlMeansDenoisingColored(cv2.imread(IMG_PATH, 1), None, 10, 10, 7, 21)

    #  結果を出力
    cv2.imwrite(bufs[0] + '_50_gray' + bufs[1], gray)
    cv2.imwrite(bufs[0] + '_51_median3' + bufs[1], img_median3)
    cv2.imwrite(bufs[0] + '_52_median5' + bufs[1], img_median5)
    cv2.imwrite(bufs[0] + '_53_bilateral1' + bufs[1], img_bilateral1)
    cv2.imwrite(bufs[0] + '_53_bilateral2' + bufs[1], img_bilateral2)
    cv2.imwrite(bufs[0] + '_53_bilateral3' + bufs[1], img_bilateral3)
    cv2.imwrite(bufs[0] + '_53_bilateral4' + bufs[1], img_bilateral4)
    cv2.imwrite(bufs[0] + '_53_bilateral5' + bufs[1], img_bilateral5)
    cv2.imwrite(bufs[0] + '_54_nlm1' + bufs[1], img_nlm1)

if __name__ == "__main__":
    main()

# EOF #