from utilities.rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import _pickle as pickle
# Define paths
filename = 'models/all_images_finalized_model_last.pkl'

path_to_images =input("Enter the test images path : ")      #images_1          test_images
path_to_masks =input("Enter the segmented test images path : ")         #images_1_mask   test_mask
path_to_feature="features"
# Grab the image and mask paths
image_paths = sorted(glob.glob(path_to_images + "/*.jpg"))
mask_paths = sorted(glob.glob(path_to_masks + "/*.jpg"))

loaded_model ,le,descriptor= pickle.load(open(filename, 'rb'))
print(loaded_model)
properties=['Peepal','Ashok','Neem','Tulsi','Bargad','Guava','Madagascar','chinese','Kapok','Mango','River']
# Loop over a sample of the images
for i in np.random.choice(np.arange(0, len(image_paths)), 10):
    # Grab the image and mask paths
    image_path = image_paths[i]
    mask_path = mask_paths[i]
    # Load the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
        
    # Describe the image
    features = descriptor.describe(image, mask)

    # Predict what type of flower the image is
    flower = le.inverse_transform(loaded_model.predict([features]))[0]
    print(image_path)
    #print("Prediction: {}".format(flower.upper()))
    for j in range(0,len(properties)):
            if properties[j] in flower :
                    print("Prediction: {}".format(properties[j].upper()))
                    file='features/'+properties[j]
                    with open(file,'r') as fp:
                            print(2*"\n")
                            print(fp.read())
    
    
    plt.imshow(image);
    plt.show();
    cv2.waitKey(0);
