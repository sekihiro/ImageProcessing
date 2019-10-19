#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import cv2
import pprint
import numpy as np
import time

# ---------- 不審物判定 ----------
# フレーム差分法

TMP_PATH = '/tmp/'

#  不審物判定の閾値
MIN_MOMENT = 50000

#  フレーム差分の計算
def frame_sub(img1, img2, img3, th):

    #  フレームの絶対差分
    # 直前のフレームと比較して画素に変化があれば、その差分の大きさが大きいほど白くなる
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img2, img3)

    # 2 つの差分画像の論理積
    diff = cv2.bitwise_and(diff1, diff2)

    #  二値化処理
    diff[diff < th] = 0
    diff[diff >= th] = 255

    #  メディアンフィルタ処理（ゴマ塩ノイズ除去）
    mask = cv2.medianBlur(diff, 5)

    return diff

def main():

    #  カメラのキャプチャ
    cap = cv2.VideoCapture(0)

    #  フレームを 3 枚取得してグレースケール変換
    frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

    #  カウント変数の初期化
    cnt = 0

    while(cap.isOpened()):

        try:

            #  フレーム間差分を計算
            mask = frame_sub(frame1, frame2, frame3, th=10)

            #  白色領域のピクセル数を算出
            moment = cv2.countNonZero(mask)
            print('moment = {}'.format(moment))

            #  白色領域のピクセル数が一定以上なら不審物有りと判定
            if moment > int(MIN_MOMENT):

                print(" 不審物を検出しました： ", cnt)

                filename = TMP_PATH + "frame" + str(cnt) + ".jpg"

                #cv2.imwrite(filename, frame2)

                cnt += 1

            #  結果を表示q
            #cv2.imshow("Frame2", frame2)
            #cv2.imshow("Mask", mask)

            #time.sleep(10)

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