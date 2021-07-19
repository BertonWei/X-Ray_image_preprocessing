import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import os
yourPath = 'C:/Hospital_Project/Image_preprocessing/images/DICOM/'
def read_xray(path, voi_lut = True, fix_monochrome = True):
    input_dicom = pydicom.read_file(path)
    print(input_dicom.pixel_array.shape)
    # VOI LUT 
    if voi_lut:
        image = apply_voi_lut(input_dicom.pixel_array, input_dicom)
    else:
        image = input_dicom.pixel_array
    # MONOCHROME
    if fix_monochrome and input_dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.amax(image) - image
    image = image - np.min(image)
    print(image.shape)
    return image
def image_16bit_to_8bit(type_16bit):
    min_16bit = np.min(type_16bit)
    max_16bit = np.max(type_16bit)
    image_8bit = np.array(np.rint(255 * ((type_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit

allFileList = os.listdir(yourPath)
for file in allFileList:
  if os.path.isdir(os.path.join(yourPath,file)):
    print("I'm a directory: " + file)

  elif os.path.isfile(yourPath+file):
    img = read_xray(yourPath+file)
    img=image_16bit_to_8bit(img)
    cv2.imwrite(file[:-4]+'.png', img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
