#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np

# ---------- 動画ファイル、及びUSBカメラ画像の読み込み ----------

MP4_PATH = '/home/seki/book/ImageProcessingAlgorithm/3-10/input.mp4'

def main():

    # VideoCapture オブジェクトを取得
    cap = cv2.VideoCapture(MP4_PATH)    # ファイル指定
    #cap = cv2.VideoCapture(0)  # USBカメラ

    print('WIDTH : {}'.format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('HEIGHT : {}'.format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame Per Sec : {}'.format(cap.get(cv2.CAP_PROP_FPS)))
    print('Frame Count : {}'.format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('Play Time [sec] : {}'.format(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)))
    print("..... Type 'q' to quit .....")

    # 動画終了まで繰り返し
    while(cap.isOpened()):

        try:

            # 1コマ分のキャプチャ画像データを読み込み
            # readすると次のフレームに進む
            ret, frame = cap.read()

            # ガウシアンフィルタ処理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=30)

            # フレームを表示
            cv2.imshow("Flame", frame)
            cv2.imshow("Flame2", gaussian) 

            # qキーが押されたら終了
            # waitKey()は引数に指定した時間(単位はミリ秒)動作を止めてキーボード入力を待つ関数
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        except:
            print('except')
            break

    # 動画ファイル閉じる、もしくはキャプチャデバイス終了
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# EOF #