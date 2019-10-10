#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
import os
import cv2

IMG_PATH = '/home/seki/sample_data/animal.jpg'

def main():

    # 入力画像の読み込み
    if not os.path.isfile(IMG_PATH):
        print("\n\n" + IMG_PATH + ' is not founded \n\n')
        sys.exit(1)

    img = cv2.imread(IMG_PATH)

    # グレースケール変換
    gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 結果を出力
    #cv2.imwrite("gray2.jpg", gray2)
    cv2.imshow("gray2", gray2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# EOF #