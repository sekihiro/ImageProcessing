#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- コーナー検出 ----------
# Harris
# J.Shi and C.Tomasi
# FAST(Features from Accelerated Segment Test)

IMG_PATH = '/home/seki/sample_data/chess.png'

def main():

    # 画像を読み込み、グレースケール変換
    # コーナー検出ではグレースケールの画像で特徴を取り出す
    chess_board = cv2.imread(IMG_PATH)
    gray_chess_board = cv2.cvtColor(chess_board, cv2.COLOR_BGR2GRAY)
    corner_harris = chess_board.copy()
    corner_shi = chess_board.copy()
    corner_fast = gray_chess_board.copy()

    # --- Harrisコーナー検出 ---
    # いろいろな方向に画素の位置を元の位置(x,y)から微小に(u,v)移動してみて画素値がどのように違うかを求める
    # 単精度浮動小数点型にキャスト
    gray = np.float32(gray_chess_board)
    # blocksize：コーナー検出の際に考慮する隣接する領域のサイズ
    # ksize：ソーベルフィルタの勾配パラメーターのカーネルサイズ
    # k：コーナーを判定する閾値
    dst = cv2.cornerHarris(src = gray, blockSize = 2, ksize = 3, k = 0.04)
    # 膨張処理を施して特徴づけ
    # マーキングのために行っているだけで、特にコーナー検出と直接関係ない
    dst = cv2.dilate(dst, None)
    # 元画像に、先ほど処理したdstが0.01*dst.max()より大きい値の部分を青色で表示するように値を代入
    corner_harris[dst > 0.01 * dst.max()] = [255, 0, 0]

    # --- J.ShiとC.Tomasiのコーナー検出 ---
    # Harrisを改良(簡素化)し、より良い結果を出すようにした
    # goodFeaturesToTrack()でコーナを検出
    # 第２引数：検出したいコーナー数を指定。−1にすると全てのコーナーを検出する。
    # 第３引数：検出するコーナーの最低限の質を0から1の間の値で指定。
    # 第４引数：検出される2つのコーナー間の最低限のユークリッド距離。
    corners = cv2.goodFeaturesToTrack(gray_chess_board, 100, 0.01, 10)
    # np.int0で整数にキャスト
    corners = np.int0(corners)
    # for-in文を使って、cornersからコーナーデータを一つずつ取り出す
    # これをravel()で１次元化して(x, y)の座標に代入
    # この(x, y)を用いて、circle()を使ってコーナー部分に丸印をつける。半径3、太さを−1にして塗りつぶしの指定。
    for i in corners:
        (x, y) = i.ravel()
        cv2.circle(corner_shi, (x, y), 3, color = (0, 255, 0), thickness = -1)

    # --- FASTのコーナー検出 ---
    # 注目画素pの周辺の円周上の決まった16画素を観測し、輝度がしきい値以上のピクセルが連続してn個以上になる点を特徴点とする
    fast = cv2.FastFeatureDetector_create()
    #fast.setThreshold(150)
    #fast.setNonmaxSuppression(False)
    kp = fast.detect(corner_fast, None)
    corner_fast2 = cv2.drawKeypoints(corner_fast, kp, None)
    print('print FAST params:')
    print("\tThreshold: ", fast.getThreshold())
    print("\tnonmaxSuppression: ", fast.getNonmaxSuppression())
    print("\tneighborhood: ", fast.getType())
    print("\tTotal Keypoints with nonmaxSuppression: ", len(kp))

    # 結果を出力
    cv2.imshow('chess_board', chess_board)
    cv2.imshow('Harris', corner_harris)
    cv2.imshow('Shi-Tomasi', corner_shi)
    cv2.imshow('FAST', corner_fast2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# EOF #