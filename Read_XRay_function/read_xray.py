import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
#read_xray function
def read_xray(path, voi_lut = True, fix_monochrome = True):
    input_dicom = pydicom.read_file(path)
    # VOI LUT 
    if voi_lut:
        image = apply_voi_lut(input_dicom.pixel_array, input_dicom)
    else:
        image = input_dicom.pixel_array
    # MONOCHROME
    if fix_monochrome and input_dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.amax(image) - image
    image = image - np.min(image)
    return image