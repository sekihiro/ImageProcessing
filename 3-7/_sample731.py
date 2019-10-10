#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import numpy as np

# ---------- 幾何変換 ----------
# アフィン変換 (変換行列の2 x 3部分を使った変換)
# 射影変換 (3 x 3すべてを使うより一般的な変換) [別名：透視変換、ホモグラフィ変換]
# https://note.nkmk.me/python-opencv-warp-affine-perspective/


IMG_PATH = '/home/seki/sample_data/city.jpg'

THETA = 45      #  回転角
theta_rad = (THETA / 180.0) * np.pi
SCALE = 1.0     #  拡大率

def main():

    #  入力画像をグレースケールで読み込み
    gray = cv2.imread(IMG_PATH, 0)
    bufs = os.path.splitext(IMG_PATH)

    # 入力画像サイズ
    (h, w) = int(gray.shape[0]), int(gray.shape[1])

    # 入力画像の中心座標
    (oy, ox) = int(gray.shape[0]/2), int(gray.shape[1]/2)

    # 回転後の画像サイズを計算
    w_rot = int(np.round(h * np.absolute(np.sin(theta_rad)) + w * np.absolute(np.cos(theta_rad))))
    h_rot = int(np.round(h * np.absolute(np.cos(theta_rad)) + w * np.absolute(np.sin(theta_rad))))
    size_rot = (w_rot, h_rot)

    # ----- 回転変換行列の算出 ----- #
    # 画像左上中心で回転
    R_00 = cv2.getRotationMatrix2D((0, 0), THETA, SCALE)
    # 画像中心で回転
    R_center = cv2.getRotationMatrix2D((ox, oy), THETA, SCALE)
    # 画像中心で回転 + 平行移動 (回転後画像がはみ出さないように)
    Transe_center = R_center.copy()
    Transe_center[0][2] = Transe_center[0][2] - w / 2 + w_rot / 2
    Transe_center[1][2] = Transe_center[1][2] - h / 2 + h_rot / 2

    # ----- 射影変換行列の生成 ----- #
    # 変換前の4点の座標 (NumPy配列)
    pts1 = np.float32([ [[0, 0], [0, h], [w, h], [w, 0]] ])
    # 変換後の4点の座標 (NumPy配列)
    pts2 = np.float32( [[20, 50], [50, 175], [300, 205], [380, 20]] )
    PT = cv2.getPerspectiveTransform(pts1, pts2)

    #  アフィン変換
    affine_00 = cv2.warpAffine(gray, R_00, gray.shape, flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    affine_center = cv2.warpAffine(gray, R_center, gray.shape, flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    affine_center_resize = cv2.warpAffine(gray, Transe_center, size_rot, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

    # 射影変換
    perspect = cv2.warpPerspective(gray, PT, (w, h))

    #  結果を出力
    cv2.imwrite(bufs[0] + '_00_gray' + bufs[1], gray)
    cv2.imwrite(bufs[0] + '_01_affine_00' + bufs[1], affine_00)
    cv2.imwrite(bufs[0] + '_01_affine_center' + bufs[1], affine_center)
    cv2.imwrite(bufs[0] + '_01_affine_center_resize' + bufs[1], affine_center_resize)
    cv2.imwrite(bufs[0] + '_02_perspect' + bufs[1], perspect)


if __name__ == "__main__":
    main()

# EOF #