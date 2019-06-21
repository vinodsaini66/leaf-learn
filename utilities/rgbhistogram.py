import cv2
import numpy as np
import matplotlib.pyplot as plt


class RGBHistogram:
	def __init__(self, bins):
		# Store the number of bins the histogram will use
		self.bins = bins

	def describe(self, image, mask=None):
		# Compute a 3D histogram in the RGB colorspace, then normalize the histogram so that images
		# with the same content will have roughly the same histogram
		mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
		
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
		
		cv2.normalize(hist, hist)
		return (hist.flatten())

