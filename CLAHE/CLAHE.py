import cv2 
  
# Reading the image from the present directory 
image = cv2.imread("path") 
# Resizing the image for compatibility 
image = cv2.resize(image, (500, 600)) 
  
# The initial processing of the image 
# image = cv2.medianBlur(image, 3) 
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# The declaration of CLAHE  
# clipLimit -> Threshold for contrast limiting 
clahe = cv2.createCLAHE(clipLimit = 10,tileGridSize=(10,10)) 
final_img = clahe.apply(image_bw)
cv2.imwrite('output.png', final_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])