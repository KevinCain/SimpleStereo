import sys
import os

import numpy as np
import cv2

import simplestereo as ss
"""
Rectify a couple of images using a RectifiedStereoRig
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
loadFile = os.path.join(curPath,"revok","rigRect.json")      # StereoRig file

# Image paths
img_dir = "X:\\data\\chessboard_e\\subjects\\"

# Load stereo rig from file
rigRect = ss.RectifiedStereoRig.fromFile(loadFile)

# Read right and left image (please ensure the order!!!)
#img1 = cv2.imread(os.path.join(img_dir,'bath_L.pgm'))
#img2 = cv2.imread(os.path.join(img_dir,'bath_C.pgm'))
#img1 = cv2.imread(os.path.join(img_dir,'door_L.pgm'))
#img2 = cv2.imread(os.path.join(img_dir,'door_C.pgm'))
img1 = cv2.imread(os.path.join(img_dir,'bath_C.pgm'))
img2 = cv2.imread(os.path.join(img_dir,'bath_rgb.jpg'))

# Optional
rigRect.computeRectificationMaps(alpha=0) # Alpha=0 may help removing unwanted border

# Simply rectify two images (it takes care of distortion too)
img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)

# Resize here for displaying purposes only, *after* drawing the lines
resized_img2 = cv2.resize(img2_rect, (1024, 1024)) 
cv2.imshow('Rectified center', img1_rect)
cv2.imshow('Rectified rgb', resized_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
