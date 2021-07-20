import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import os
import warnings 
warnings.filterwarnings('ignore',category=FutureWarning) #because of numpy version
from tensorflow.keras import backend as K
K.clear_session()
from keras.models import  Model, Input
from keras.layers import Conv2D, Dense, MaxPooling2D, SeparableConv2D, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D,Flatten,Average, Dropout
from keras.layers import Add, Activation, Dropout, Flatten, Dense, Lambda, LeakyReLU, PReLU
from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef, mean_squared_error, mean_squared_log_error
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, accuracy_score, log_loss
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras import backend as K
import imageio
yourPath = './DICOM/'
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
def resnet_bs(num_filters=64, num_res_blocks=16, res_block_scaling=None):
    x_in = Input(shape=(512,512,1))
    x = b = Conv2D(num_filters, (3, 3), padding='same')(x_in)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, (3, 3), padding='same')(b)
    x = Add()([x, b])
    x = Conv2D(1, (3, 3), padding='same')(x)
    return Model(x_in, x, name="ResNet-BS")

def res_block(x_in, filters, scaling):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x_in)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


resnet_bs = resnet_bs(num_filters=64, num_res_blocks=22, res_block_scaling=0.1)
resnet_bs.summary()
#load model:
resnet_bs.load_weights("ResNet-BS.bestjsrt4500_num_filters_64_num_res_blocks_22.h5") 
print("Loaded model from disk")
resnet_bs.summary()

allFileList = os.listdir(yourPath)
for file in allFileList:
  if os.path.isdir(os.path.join(yourPath,file)):
    print("I'm a directory: " + file)
  elif os.path.isfile(yourPath+file):
    img = read_xray(yourPath+file)
    No_normalization=image_16bit_to_8bit(img)
    #resnet_bs處理
    img = cv2.resize(No_normalization,(512,512))
    img = img.astype('float32') / 255
    x1 = np.expand_dims(img, axis=0)
    pred = resnet_bs.predict(x1)
    test_img = np.reshape(pred, (512,512,1)) 
    imageio.imwrite("{}.png".format(file[:-4]), test_img)
    #CLAHE處理
    bone_image = cv2.imread(file[:-4]+".png") 
    image_bw = cv2.cvtColor(bone_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 10,tileGridSize=(10,10)) 
    final_img = clahe.apply(image_bw)
    cv2.imwrite(file[:-4]+'.png', final_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    