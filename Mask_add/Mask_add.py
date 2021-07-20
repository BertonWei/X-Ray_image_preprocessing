import numpy as np
import cv2
import os
#mask folder
maskPath = 'E:/NEW_預處理/lung_segmentation/'
#Original image folder
OriginalPath = 'E:/NEW_預處理/images/'
#mask
segmentation = cv2.imread('path',0)
#Original image
allFileList = os.listdir(maskPath)
for file in allFileList:
  if os.path.isdir(os.path.join(maskPath,file)):
    print("I'm a directory: " + file)
  elif os.path.isfile(maskPath+file):
    print(maskPath+file)
    segmentation = cv2.imread(maskPath+file,0)
    cv2.imshow('My Image', segmentation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(OriginalPath+file)
    Bone_suppression = cv2.imread(OriginalPath+file)
    
    
    Bone_suppression = cv2.resize(Bone_suppression, (512, 512))
    image=cv2.add(Bone_suppression, np.zeros(np.shape(Bone_suppression), dtype=np.uint8), mask=segmentation)
    cv2.imwrite(file[:-4]+'.png', image,[cv2.IMWRITE_PNG_COMPRESSION, 0])
