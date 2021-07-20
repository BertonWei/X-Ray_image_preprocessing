# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 00:52:53 2021

@author: Chunayi
"""
import numpy as np
from PIL import Image
import cv2
import os
#mark folder
markPath = 'C:/Users/EN308/Desktop/chuanyi/unet-master-master/data/membrane/train/label/'
#Original image folder
OriginalPath = 'C:/Users/EN308/Desktop/BertonWei/X-Ray_image_preprocessing-main/Mask_add/add/'
#mask
segmentation = cv2.imread('path',0)
#Original image
allFileList = os.listdir(markPath)
for file in allFileList:
  if os.path.isdir(os.path.join(markPath,file)):
    print("I'm a directory: " + file)
  elif os.path.isfile(markPath+file):
    print(OriginalPath+file)

    image = cv2.imread(OriginalPath+file[:-4]+'.png')
    mask = cv2.imread(markPath+file,cv2.IMREAD_GRAYSCALE)
    ret, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    redImg = np.zeros(image.shape, image.dtype)
    redImg[:,:] = (0, 0, 255)
    redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
    result = cv2.addWeighted(redMask, 1, image, 1, 0, image)
    # 顯示圖片
    cv2.imshow('result', result)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(file[:-4]+'.png', image,[cv2.IMWRITE_PNG_COMPRESSION, 0])
