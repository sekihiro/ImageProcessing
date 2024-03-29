#!/usr/bin/env python
#-*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

# グレースケール化 #

IMG_PATH = '/home/seki/sample_data/dog.jpg'

def main():
    #  入力画像を読み込み
    img = cv2.imread(IMG_PATH)

    #  グレースケール変換
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #  方法 1(NumPy でヒストグラムの算出 )
    #hist, bins = np.histogram(gray.ravel(),256,[0,256])

    #  方法 2(OpenCV でヒストグラムの算出 )
    hist = cv2.calcHist([img],[0],None,[256],[0,256])

    #  ヒストグラムの中身表示
    print(hist)

    #  グラフの作成
    plt.xlim(0, 255)
    plt.plot(hist)
    plt.xlabel("Pixel value", fontsize=20)
    plt.ylabel("Number of pixels", fontsize=20)
    plt.grid()
    plt.show()

    
if __name__ == "__main__":
    main()

# EOF #