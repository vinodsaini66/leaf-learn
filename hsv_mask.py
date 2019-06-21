import numpy as np 
import cv2 
from matplotlib import pyplot as plt
import os
### Image operation using thresholding
listoffiles=os.listdir("test")

for i in range(0,len(listoffiles)) :             
               img = cv2.imread("test/"+listoffiles[i])
               hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
               lower_green = np.array([20,20,20])
               upper_green = np.array([80,255,255])
               mask = cv2.inRange(hsv, lower_green, upper_green)
               cv2.imwrite("mask/"+listoffiles[i],mask)
               print(i)



