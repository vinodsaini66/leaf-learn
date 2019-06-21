import numpy as np                      
import cv2 
from matplotlib import pyplot as plt
import os
from utilities.rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import glob
import _pickle as pickle
### Image

img_url=input("Enter the image url :")
img_name=input("Enter the image Name :")

#Resize Original image

img = cv2.imread(img_url)
dim = (800, 600)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img_save=img_name+".jpg"
cv2.imwrite("test/"+img_save, resized)


#Segmentation of new image
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
lower_green = np.array([20,20,20])
upper_green = np.array([80,255,255])
mask = cv2.inRange(hsv, lower_green, upper_green)
cv2.imwrite("mask/"+img_save,mask)

#Define path of models and images
filename = 'models/test_images_finalized_model_last.pkl'
image_path = "test/"+img_save       #images_1          test_images
mask_path = "mask/"+img_save     #images_1_mask   test_mask
path_to_feature="features"

loaded_model ,le,descriptor= pickle.load(open(filename, 'rb'))
print(loaded_model)
properties=['Peepal','Ashok','Neem','Tulsi','Bargad','Guava','Madagascar','chinese','Kapok','Mango','River']


image =cv2.imread(image_path)
mask = cv2.imread(mask_path)


features = descriptor.describe(image, mask)

    # Predict what type of flower the image is
flower = le.inverse_transform(loaded_model.predict([features]))[0]
print(image_path)
check=image_path.split("/")[1].split(".")[0]
print(check)
print("Prediction: {}".format(flower.upper()))
for j in range(0,len(properties)):
            if properties[j] in flower :
                    file='features/'+properties[j]
                    with open(file,'r') as fp:
                            print(2*"\n")
                            print(fp.read())

plt.imshow(image);
plt.show();
cv2.waitKey(0);
















