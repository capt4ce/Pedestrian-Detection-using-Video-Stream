# Stream Video with OpenCV from an Android running IP Webcam (https://play.google.com/store/apps/details?id=com.pas.webcam)
# Code Adopted from http://stackoverflow.com/questions/21702477/how-to-parse-mjpeg-http-stream-from-ip-camera

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import urllib2
import sys

host = "192.168.43.1:8080"
# host = "172.16.164.179:8080"

if len(sys.argv)>1:
    host = sys.argv[1]

hoststr = 'http://' + host + '/video'
print('Streaming ' + hoststr)

stream=urllib2.urlopen(hoststr)

# initialize the HOG (Histogram Oriented Gradients) descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

bytes=''
while True:
	status = "No Targets"
	bytes+=stream.read(1024)
	a = bytes.find('\xff\xd8')
	b = bytes.find('\xff\xd9')
	if a!=-1 and b!=-1:
		jpg = bytes[a:b+2]
		bytes= bytes[b+2:]
		image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),1)#cv2.CV_LOAD_IMAGE_COLOR)
		image = imutils.resize(image, width=min(400, image.shape[1]))

		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

		# show the frame and record if a key is pressed
		cv2.imshow("Frame", image)
		key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
