# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:39:43 2021

@author: Chunayi
"""

import cv2
img = cv2.imread('1.png')
print (img.shape)
center = img.shape
print(center[1])
print(center[0])
center_x=center[1] /2
center_y=center[0] /2
if center[1]<center[0]:
    w=center[1]
    h=center[1]
else:
    w=center[0]
    h=center[0]
x = center_x - w/2
y = center_y - h/2 
crop_img = img[int(y):int(y+h), int(x):int(x+w)]

cv2.imshow('Image', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

