import sys
import os

import cv2

import simplestereo as ss
"""
Remove lens distortion from images using StereoRig
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
loadFile = os.path.join(curPath, "revok", "rig.json")

# Image paths
#img_dir_L = "X:\\data\\chessboard_e\\extrinsics_sync\\left\\"
#img_dir_R = "X:\\data\\chessboard_e\\extrinsics_sync\\center\\"
img_dir = "X:\\data\\chessboard_e\\subjects\\"


# Load stereo rig from file
rig = ss.StereoRig.fromFile(loadFile)

# Read right and left image (please ensure the order!!!)
#img1 = cv2.imread(os.path.join(img_dir_L,'left (13).jpg'))
#img2 = cv2.imread(os.path.join(img_dir_R,'center (13).jpg'))
img1 = cv2.imread(os.path.join(img_dir,'ad_C.pgm'))
img2 = cv2.imread(os.path.join(img_dir,'ad_rgb.jpg'))

# Simply undistort two images
img1, img2 = rig.undistortImages(img1, img2)

# Show images
# Resize here for displaying purposes only, *after* drawing the lines
resized_img2 = cv2.resize(img2, (1024, 1024)) 

cv2.imshow('img1 Undistorted', img1)
cv2.imshow('img2 Undistorted', resized_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done!")
