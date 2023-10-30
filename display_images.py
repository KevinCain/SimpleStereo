import sys
import os

import cv2
import numpy as np

import simplestereo as ss
"""
Remove lens distortion from images using StereoRig
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join(curPath,"revok","center")

# StereoRig file
loadFile = os.path.join(curPath,"revok","rig.json")

# Load stereo rig from file
rig = ss.StereoRig.fromFile(loadFile)

# Read right and left image (please ensure the order!!!)
img1 = cv2.imread(os.path.join(imgPath,'left (1).pgm'))
img2 = cv2.imread(os.path.join(imgPath,'center (1).pgm'))


# Simply undistort two images
img1, img2 = rig.undistortImages(img1, img2)

# Your 3x3 fundamental matrix
F = rig.getFundamentalMatrix()

# Number of evenly spaced epipolar lines
N = 10

# Height and width of img1
height1, width1 = img1.shape[0], img1.shape[1]

# Compute the y-coordinates at which the lines will be drawn
y_coords = np.linspace(0, height1, N, endpoint=False)

# Choose the x-coordinate as the midpoint of the width of img1 for all points
x_coord = width1 // 2

# Create the list of points
x1_points = [(x_coord, int(y)) for y in y_coords]

# Call the drawCorrespondingEpipolarLines function
ss.utils.drawCorrespondingEpipolarLines(img1, img2, F, x1=x1_points, x2=[], color=(255, 155, 0), thickness=2)

# Resize here for displaying purposes only, *after* drawing the lines
#resized_img1_u = cv2.resize(img1, (512, 512)) 
#resized_img2_u = cv2.resize(img2, (512, 512))

# Show images
cv2.imshow('img1 Undistorted', img1)
cv2.imshow('img2 Undistorted', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done!")
