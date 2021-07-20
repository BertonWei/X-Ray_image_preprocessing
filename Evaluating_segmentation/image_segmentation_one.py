# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:03:23 2021
https://www.jeremyjordan.me/evaluating-image-segmentation-models/?fbclid=IwAR0FwxrbsFfup81yDehI1GwxoyxZHYCygX664ZOB65hNWNS9s_KHjL6SwEg
@author: Chunayi
"""
import os
import numpy as np
from PIL import Image
import cv2




#ground_truth array 
ground_truth = cv2.imread("E:/unet-master-master/data/membrane/train/036-marked.jpg",cv2.IMREAD_GRAYSCALE)
ret, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
# 顯示圖片
cv2.imshow('ground_truth', ground_truth)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
#prediction array 
prediction = cv2.imread("C:/Users/Chunayi/Desktop/13_predict.jpg",cv2.IMREAD_GRAYSCALE)
ret, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
# 顯示圖片
cv2.imshow('prediction', prediction)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
#zero array 
ground_truth_shap = ground_truth.shape
zreo_array =np.zeros((ground_truth_shap[0], ground_truth_shap[1]))

#ground_truth
ground_truth = np.logical_or(ground_truth, zreo_array)
#prediction
prediction = np.logical_or(prediction, zreo_array)
#intersection
intersection = np.logical_and(ground_truth, prediction)
#union
union = np.logical_or(ground_truth, prediction)


#IoU calculation formula
iou_score = np.sum(intersection) / np.sum(union)
print('IoU : '+str(iou_score))
#reall calculation formula
reall = np.sum(intersection) / np.sum(ground_truth)
print('reall : '+str(reall))
#Precision calculation formula
Precision = np.sum(intersection) / np.sum(prediction)
print('Precision : '+str(Precision))


