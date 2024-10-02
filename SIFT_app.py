#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

	def __init__(self):


		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self._cam_id = 0
		self._cam_fps = 10 #originally 10, didnt change anything
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv2.VideoCapture("/home/fizzer/Downloads/Leopard.mp4") #CHANGE THIS FOR NEW VIDEO
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)

		print("Loaded template image file: " + self.template_path)

	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)




	def SLOT_query_camera(self): #RUN SIFT HERE
		#load image
		imgbad = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(imgbad, (320, 240))  # Adjust width and height here
	

		#create sift object
		sift = cv2.SIFT_create()

		#detect and compute keypoints in image
		kp_image, desc_image = sift.detectAndCompute(img,None) 



		index_params = dict(algorithm=0, trees = 2) #sets parameters for FLANN, we are using algorithm 0 (KD-Tree Algo) with 5 trees
			#more trees increases accuracy at expense of more compute
				#however in my testing changing number of trees didnt change the FPS
		search_params = dict() #we are using default search params for FLANN
		flann = cv2.FlannBasedMatcher(index_params, search_params) #creates the FLANN matching algorithm object


		#create video frame from camera
		ret, frame = self._camera_device.read()
		#frame = cv2.resize(framebad, (320, 240))  # Adjust width and height here

		#put video frame into greyscale and then find keypoints
		grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)



		matches = flann.knnMatch(desc_image,desc_grayframe,k=2) #finds the k nearest neighbors according to flann algorithm
		#in english, finds the most similar point in the second image compared with descriptors from first image

		#restrict number of matches to within a small distance (distance is essentially how accurate the match is)
		#Called "Lowes Ratio Test"
		good_matches = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				good_matches.append(m)
			#matches is a list of pairs (tuples), where each pair consists of two descriptors:
			#m is the closest (first) match for a descriptor in the first image.
    		#n is the second closest (second) match.
			# we check the closest match is significantly closer than the second, then we know its better 


		#draw matches on the image
		img3 = cv2.drawMatches(img, kp_image,grayframe,kp_grayframe,good_matches,None)


		#Homography (finding the image within the other image)
		if len(good_matches) > 10: #check if at a certain time we have enough good matches


			#creates arrays of keypoints from both image and video frame
			query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
			train_pts =np.float32([kp_grayframe[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

			#calculates transformation matrix using RANSAC with max distance threshold of 5
			matrix, mask = cv2.findHomography(query_pts,train_pts,cv2.RANSAC,5.0)
			matches_mask = mask.ravel().tolist() #flattens mask array to 1D

			#Perspective Transform
			#define corners of the image
			h,w = img.shape
			pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,matrix)
				#maps homography matrix to image within box

			#draw the outline of the image
			homography = cv2.polylines(frame,[np.int32(dst)], True, (255,0,0),3)
		
		else: #if there isnt a good enough match, just display original image
			homography = frame


		#QUESTION: HOW CAN I MAKE IT RUN FASTER? FPS DOESNT CHANGE ANYTHING

		pixmap = self.convert_cv_to_pixmap(homography)
		self.live_image_label.setPixmap(pixmap)




	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())

