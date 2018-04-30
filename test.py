# import the necessary packages
from sklearn.externals import joblib
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
from sklearn import svm
from sklearn.externals import joblib
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import os
from sklearn import svm
import time


# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()

ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())


(winW, winH) = (64, 64)

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


#accessing saved file
svc = joblib.load('trained_shape_detector.pkl') 

# loop over the test dataset
for imagePath in paths.list_images(args["test"]):
	# load the test image, convert it to grayscale, and resize it 
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(500, image.shape[1]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	for resized in pyramid(gray, scale=1.25):
		# loop over the sliding window for each layer of the pyramid
		clone=resized.copy()
		for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			#pic = cv2.resize(window, (64, 128))
			H = feature.hog(window, orientations=9, pixels_per_cell=(10, 10),
				    	cells_per_block=(2, 2), transform_sqrt=True)
			pred=svc.predict(H.reshape(1,-1))[0]
			#print pred.title()    
			#clone=resized.copy()
			
			if pred.title() == 'Hex' :
				print 'triangle'
   				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
   			
			cv2.imshow("Image", clone)
			'''
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.putText(clone, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
				(0, 255, 0), 3)
			'''
			cv2.waitKey(1)
			#time.sleep(0.0025)
	# extract Histogram of Oriented Gradients from the test image and predict the digit
	# visualize the HOG imag
	'''
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
	cv2.imshow("HOG Image", hogImage)
	
	# draw the prediction on the test image and display it
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)
	#image =cv2.resize(image, (200, 150))
	cv2.imshow("Test Image", image)
	cv2.waitKey(0)
	'''