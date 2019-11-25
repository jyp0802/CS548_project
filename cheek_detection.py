# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#print('predictor = {}'.format(predictor))
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
print(4)
print(type(image))
print(image.shape)
print(5)
image = imutils.resize(image, width=500)
#image = imutils.resize(image, width=128)

print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(1)
# detect faces in the grayscale image
rects = detector(gray, 1)
print('rects is {}'.format(rects))
print(2)

target = [ 'mouth' , 'right_eye' , 'nose']

mouth_loc =(0,0)
eye_loc = (0,0)
nose_loc = (0,0)
patch_center = (0,0)
# loop over the face detections
print(10)
for (i, rect) in enumerate(rects):
	print('rects in for loop = {}'.format(rect))
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# loop over the face parts individually
	print(3)

	#print(face_utils.FACIAL_LANDMARKS_IDXS.items())



	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		clone = image.copy()
		# cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# cv2.circle(clone, (i, j), 1, (200, 200, 200), 5)
		print(4)

		if name in target:
			temp = (0,0)
			point_num = len(shape[i:j])
			print('length of shape point is {}'.format(len(shape[i:j])))

			for (x, y) in shape[i:j]:
				
				temp = (temp[0]+ x, temp[1]+y)
				print('sum of xy = {}'.format(temp))
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
				#print(x,y)
				print(5)

				if name is 'mouth':
					mouth_loc = (int(temp[0]/point_num), int(temp[1]/point_num))

				elif name is 'right_eye':
					eye_loc = (int(temp[0]/point_num) , int(temp[1]/point_num))

				elif name is 'nose':
					nose_loc = (int(temp[0]/point_num) , int(temp[1]/point_num))
		else:
			pass

 

		x_dif = nose_loc[0] - eye_loc[0]
		y_dif = nose_loc[1] - eye_loc[1]

		patch_center = (int(eye_loc[0]-x_dif*0.4) , int(nose_loc[1] - int(y_dif*0.1)))
		# cv2.circle(clone, (mouth_loc[0], mouth_loc[1]), 1, (255, 0, 0), 5)
		# cv2.circle(clone, (eye_loc[0], eye_loc[1]), 1, (0, 255, 0), 5)
		# cv2.circle(clone, (nose_loc[0], nose_loc[1]), 1, (0, 0, 255), 5)	
		cv2.circle(clone, (patch_center[0], patch_center[1]), 1, (255, 255, 255), int(y_dif*0.6))
		# show the particular face part
		#cv2.imshow("ROI", roi)
		cv2.imshow("Image", clone)
		# print(7)
		cv2.waitKey(0)


	#print(patch_center)
	# visualize all facial landmarks with a transparent overlay
	#output = face_utils.visualize_facial_landmarks(image, shape)
	#cv2.imshow("Image", output)
	print(8)
	cv2.waitKey(0)