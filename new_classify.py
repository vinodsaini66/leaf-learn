from utilities.rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
import glob
import cv2
import matplotlib.pyplot as plt
import _pickle as pickle
# Define paths
path_to_images = 'test_images'
path_to_masks = 'test_mask'

# Grab the image and mask paths
image_paths = sorted(glob.glob(path_to_images + "/*.jpg"))
mask_paths = sorted(glob.glob(path_to_masks + "/*.jpg"))


path_to_images1 = 'images_1'
path_to_masks1 = 'images_1_mask'

# Grab the image and mask paths
image_paths1 = sorted(glob.glob(path_to_images + "/*.jpg"))
mask_paths1 = sorted(glob.glob(path_to_masks + "/*.jpg"))

# Initialize the list of data and class label targets
data = []
target = []

# Initialize the image descriptor
descriptor = RGBHistogram([8, 8, 8])
k=0
# Loop over the image and mask paths
for (image_path, mask_path) in zip(image_paths, mask_paths):
	# Load the image and mask
	image = cv2.imread(image_path)
	mask = cv2.imread(mask_path)
	
	#mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# Describe the image
	features = descriptor.describe(image, mask)
	# Update the list of data and targets
	data.append(features)
	target.append(image_path.split("/")[1].split(".")[0])
	
	

for (image_path, mask_path) in zip(image_paths1, mask_paths1):
	# Load the image and mask
	image = cv2.imread(image_path)
	mask = cv2.imread(mask_path)
	
	#mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# Describe the image
	features = descriptor.describe(image, mask)
	# Update the list of data and targets
	data.append(features)
	target.append(image_path.split("/")[1].split(".")[0])




# Grab the unique target names and encode the labels
target_names = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)
# Construct the training and testing splits
(train_x, test_x, train_y, test_y) = train_test_split(data, target, test_size=0.3, random_state=42)
# Train the classifier
model = RandomForestClassifier(n_estimators=25, random_state=84)

model.fit(data, target)
########################################################
tuple_objects = (model, le,descriptor)
filename = 'models/all_images_finalized_model_last.pkl'
pickle.dump(tuple_objects, open(filename, 'wb'))
####################################################################
# Evaluate the classifier


print("model saved")
# Loop over a sample of the images
