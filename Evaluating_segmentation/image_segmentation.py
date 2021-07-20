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

index=0
iou_sum=0
reall_sum=0
Precision_sum=0

# 指定要查詢的路徑
yourPath = './prediction_image/'
allFileList = os.listdir(yourPath)
for file in allFileList:
  if os.path.isfile(yourPath+file):
    
    #ground_truth array 
    ground_truth = cv2.imread("./ground_truth_image/"+file,cv2.IMREAD_GRAYSCALE)
    ret, ground_truth = cv2.threshold(ground_truth, 1, 255, cv2.THRESH_BINARY)
    # 顯示圖片
    cv2.imshow('ground_truth', ground_truth)
    
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #prediction array 
    prediction_image = cv2.imread("./prediction_image/"+file,cv2.IMREAD_GRAYSCALE)
    ret, prediction_image = cv2.threshold(prediction_image, 128, 255, cv2.THRESH_BINARY)
    # 顯示圖片
    cv2.imshow('prediction', prediction_image)
    
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #zero array 
    ground_truth_shap = ground_truth.shape
    zreo_array =np.zeros((ground_truth_shap[0], ground_truth_shap[1]))
    
    #ground_truth
    ground_truth = np.logical_or(ground_truth, zreo_array)
    #prediction
    prediction = np.logical_or(prediction_image, zreo_array)
    #intersection
    intersection = np.logical_and(ground_truth, prediction)
    #union
    union = np.logical_or(ground_truth, prediction)
    
    
    #IoU calculation formula
    iou_score = np.sum(intersection) / np.sum(union)
    #reall calculation formula
    reall = np.sum(intersection) / np.sum(ground_truth)
    #Precision calculation formula
    Precision = np.sum(intersection) / np.sum(prediction)
    
    index=index+1
    iou_sum=iou_sum+iou_score
    reall_sum=reall_sum+reall
    Precision_sum=Precision_sum+Precision
    
    #overlop set
    image = cv2.imread("./mark_image/"+file)
    redImg = np.zeros(image.shape, image.dtype)
    redImg[:,:] = (0, 0, 255)
    redMask = cv2.bitwise_and(redImg, redImg, mask=prediction_image)
    result = cv2.addWeighted(redMask, 1, image, 1, 0, image)

    cv2.imwrite('./overlap_image/'+file, result, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
#print(index)   
#print(iou_sum)   
#print(reall_sum)   
#print(Precision_sum)    

iou_average=iou_sum/index
reall_average=(reall_sum/index)*100
recision_average=(Precision_sum/index)*100
print('IoU : '+str(iou_average))   
print('reall : '+str(reall_average))   
print('Precision : '+str(recision_average))   





