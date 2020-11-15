import numpy as np
from PIL import Image
import sys
import os

def mse_(image_1, image_2):
	imageA=np.asarray(image_1)
	imageB=np.asarray(image_2)  
	err = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	return err