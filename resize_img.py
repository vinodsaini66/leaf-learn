import numpy as np 
import cv2 
from matplotlib import pyplot as plt
import os
### Image operation using thresholding
listoffiles=os.listdir("non leaf")
print(listoffiles)

for i in range(0,len(listoffiles)) :             
               img = cv2.imread("non leaf/"+listoffiles[i], cv2.IMREAD_UNCHANGED)
               dim = (128,128)
               resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
               cv2.imwrite("non leaf/"+listoffiles[i], resized)
               print(i)




