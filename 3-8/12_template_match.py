#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np

# ---------- テンプレート・マッチング ----------
# メリット：お手軽
# デメリット：柔軟性がない（拡大や縮小程度の違いでも認識できなくなる。）

IMG_PATH = '/home/seki/sample_data/fruite.png'
TEMPL_PATH = '/home/seki/sample_data/fruite_template.png'
THRESHOLD = 0.9

def main():

    #  画像を読み込み
    img = cv2.imread(IMG_PATH)

    #  画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    temp = cv2.imread(TEMPL_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    #  テンプレート画像の高さ・幅
    (h, w) = temp.shape

    # ----- テンプレート・マッチング -----
    # 入力画像に対して、テンプレート画像と同じ大きさの検索窓を左上からスライドさせながら動かしていく。
    # 移動する検索窓の各位置において、その検索窓の範囲の画像とテンプレート画像の類似度を計算する。
    # resultには検索窓を動かた際の各位置での類似度の値が入っている。
    # TM_SQDIFF             SSD (Sum of Absolute Difference)では、「画素値の差分の二乗値の和」で類似度を評価する。類似するほど値が小さい。
    # TM_CCOEFF_NORMED      ZNCC（Zero means Normalized Cross Correlation）では、「零平均正規化相互相関」と呼ばれる統計量で類似度を評価する。類似するほど値が大きい(最大１ / 最小０)
    result_ssd = cv2.matchTemplate(gray, temp, cv2.TM_SQDIFF)
    result_zncc = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)

    # 最も類似度が高い位置と低い位置を取得
    # 最も類似度の高い部分一つだけ検出する場合に使用
    # min_value     最も低い画素値
    # max_value     最も高い画素値
    # min_pt        最も低い画素値の位置座標
    # max_pt        最も高い画素値の位置座標
    (min_value, max_value, min_pt, max_pt) = cv2.minMaxLoc(result_ssd)
    print('SSD min_value: {}'.format(min_value))
    pt = min_pt
    # テンプレート・マッチングの結果を出力
    cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
    cv2.imwrite(bufs[0] + '_00_ssd' + bufs[1], img)

    # 複数検出
    loc = np.where(result_zncc >= THRESHOLD)
    img2 = cv2.imread(IMG_PATH)
    for top_left in zip(loc[1], loc[0]):
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img2, top_left, bottom_right, (0, 255, 0), 3)
    cv2.imwrite(bufs[0] + '_00_zncc' + bufs[1], img2)


if __name__ == "__main__":
    main()

# EOF #