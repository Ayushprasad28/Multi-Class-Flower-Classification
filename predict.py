# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import sys

# initialising 
dir_labels=()
dir_predict=()
num_class=0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default="training_set",
	help="path to input dataset")
ap.add_argument("-m", "--model", required=False, default="trained_model",
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

#findings the labels
for file in os.listdir(args["dataset"]) :
	temp_tuple=(file,'null')
	dir_labels=dir_labels+temp_tuple
	dir_labels=dir_labels[:-1]
	num_class=num_class+1

print("[INFO] Labels are ",dir_labels)

# load the image
print("[INFO] Loading Image...")
try :
	image = cv2.imread(args["image"])
	orig = image.copy()
except AttributeError :
	print("[INFO] Error in the test image... ")
	print('[INFO] Exiting...')
	sys.exit()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] Loading Network...")
model_base=args["model"]+'.h5'
model = load_model(model_base)

# classify the input image
dir_predict = model.predict(image)[0]
print(dir_labels)
print(dir_predict)
for i in range(num_class) :
	var = 0
	for j in range(num_class) :
		if(dir_predict[i]>=dir_predict[j]) :
			var=var+1
	if(var==num_class) :
		label=dir_labels[i]
		proba=dir_predict[i]
	elif(var==num_class-1) :
		label2=dir_labels[i]
		proba2=dir_predict[i]


label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show and save the output image
cv2.imshow("Output", output)
cv2.imwrite("Output.png",output)
cv2.waitKey(0)  #Press any key to exit the output image

print('[INFO] Exiting...')
