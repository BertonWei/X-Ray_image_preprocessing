# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 23:47:13 2021

@author: Chunayi
"""

import cv2
import os
yourPath = './PNG/'

allFileList = os.listdir(yourPath)
for file in allFileList:
  if os.path.isdir(os.path.join(yourPath,file)):
    print("I'm a directory: " + file)

  elif os.path.isfile(yourPath+file):
    # Reading the image from the present directory 
    image = cv2.imread(yourPath+file) 
    # Resizing the image for compatibility 
    image = cv2.resize(image, (500, 600)) 
    # The initial processing of the image 
    # image = cv2.medianBlur(image, 3) 
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # The declaration of CLAHE  
    # clipLimit -> Threshold for contrast limiting 
    clahe = cv2.createCLAHE(clipLimit = 10,tileGridSize=(10,10)) 
    final_img = clahe.apply(image_bw)
    cv2.imwrite(file[:-4], final_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])