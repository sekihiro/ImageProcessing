#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np

# ---------- 色検出 ----------
# 赤色を検出
# http://sh0122.hatenadiary.jp/entry/2017/10/17/220447
# https://sites.google.com/site/lifeslash7830/home/hua-xiang-chu-li/opencvniyoruhuaxiangchulisekongjian
# https://qiita.com/takaya901/items/ad6b73e1c5168b794ab7
#
# オンライン色確認ツール
# https://www.peko-step.com/tool/hsvrgb.html

MP4_PATH = '/home/seki/book/ImageProcessingAlgorithm/3-10/input.mp4'

def red_detect(img):

    # HSV 色空間に変換
    # 色相(Hue)、彩度(Saturation)、明度(Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #  赤色の HSV の値域 1
    # np.arrayの引数は、HSV
    # OpenCVの場合、H,S,Vともに256段階で保持する
    hsv_min = np.array([0, 128, 0])
    hsv_max = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    #  赤色の HSV の値域 2
    hsv_min = np.array([150, 128, 0])
    hsv_max = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色領域１と赤色領域２のＯＲ演算し、mask画像を生成
    #return mask1 + mask2
    ret_or = cv2.bitwise_or(mask1, mask2)
    return ret_or

def main():

    #  動画のキャプチャ
    cap = cv2.VideoCapture(MP4_PATH)

    while(cap.isOpened()):

        try:

            # フレームを取得
            ret, frame = cap.read()

            # 赤色検出用のmask画像 (0,1で指定された画像)
            mask = red_detect(frame)

            # 赤色検出
            # マスク画像の白部分(1の部分)のみを表示
            res = cv2.bitwise_and(frame, frame, mask = mask)

            # 結果表示
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("res", res)

            # q キーが押されたら終了
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