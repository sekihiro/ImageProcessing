#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np

# ---------- フレーム間差分法 ----------
# 移動物体を撮影した異なる時間の３枚の画像を用いて移動物体領域を取り出すことができる
# なお、３枚の画像において、背景は変化がないものとする
# ３枚の画像をABCとする
# AとB、BとCの差分画像を作成し、しきい値処理を施し２値画像ABとBCを得る
# つぎに、２値画像ABとBCのAND処理を行い、ABとBCの共通領域を取り出すことで、画像Bにおける移動物体の領域を得ることができる

MP4_PATH = '/home/seki/book/ImageProcessingAlgorithm/3-10/input.mp4'


#  フレーム差分の計算
def frame_sub(img1, img2, img3, th):

    # フレームの絶対差分
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img2, img3)

    # 2 つの差分画像の論理積
    diff = cv2.bitwise_and(diff1, diff2)

    # 単純二値化処理
    diff[diff < th] = 0
    diff[diff >= th] = 255

    #  メディアンフィルタ処理(ゴマ塩ノイズ除去)
    mask = cv2.medianBlur(diff, ksize = 3)

    return  mask

def main():

    #  動画のキャプチャ
    cap = cv2.VideoCapture(MP4_PATH)

    #  フレームを 3 枚取得してグレースケール変換
    # cap.read()[0]はリターンコード
    # readすると次のフレームに進む
    frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

    while(cap.isOpened()):

        try:

            #  フレーム間差分を計算
            mask = frame_sub(frame1, frame2, frame3, th = 30)

            #  結果を表示
            cv2.imshow("Frame2", frame2)
            cv2.imshow("Mask", mask)

            # 3 枚のフレームを更新
            frame1 = frame2
            frame2 = frame3
            frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

            # q キーが押されたら途中終了
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        except:
            print('except')
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# EOF #