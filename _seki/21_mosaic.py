#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np
import time

# ---------- モザイク処理 ----------

#  モザイク処理
def mosaic(img, alpha):

    #  画像の高さと幅
    w = img.shape[1]
    h = img.shape[0]

    #  縮小→拡大でモザイク加工
    img = cv2.resize(img,(int(w*alpha), int(h*alpha)))
    img = cv2.resize(img,(w, h), interpolation=cv2.INTER_NEAREST)
    return img

def main():

    #  カスケード型識別器の読み込み
    cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    #  カメラのキャプチャ
    cap = cv2.VideoCapture(0)

    #  動画終了まで繰り返し
    while(cap.isOpened()):

        try:

            #  フレームを取得
            ret, frame = cap.read()

            #  グレースケール変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #  顔領域の探索
            face = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

            #  顔領域を赤色の矩形で囲む
            for (x, y, w, h) in face:

                #  顔部分を切り出してモザイク処理
                frame[y:y+h, x:x+w] = mosaic(frame[y:y+h, x:x+w], 0.025)

            #  フレームを表示
            resized_frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
            cv2.imshow("Flame", resized_frame)

            # q キーが押されたら途中終了
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        except:
            import traceback
            traceback.print_exc()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# EOF #