import sys
import os

import cv2
import numpy as np

import simplestereo as ss
"""
Correct World Camera images using ML2 reported intrinsics
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
#imgPathL = os.path.join(curPath, 'revok', 'chessboard_d', 'leftb (3).pgm')
imgPathL = os.path.join(curPath, 'revok', 'center', 'center (7).pgm')
#imgPathR = os.path.join(curPath, 'revok', 'chessboard_d', 'rgbb (3).jpg')

# Read image
img = cv2.imread(imgPathL)

# Check if the image was loaded
if img is None:
    print("Image not loaded. Check the path.")
    exit()

# Image size
img_size = (img.shape[1], img.shape[0])

# Camera intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
intrinsic_matrix = np.array([[581.5074, 0, 510.7213],
                             [0, 581.4514, 505.8165],
                             [0, 0, 1]], dtype=np.float32)

# Distortion coefficients [k1, k2, p1, p2, k3]
# Note: k4 value from ML2 is not directly supported by cv2.undistort
distortion_coefficients = np.array([0.1037, -0.0731, -0.0009, 0.0007, -0.0628], dtype=np.float32)

# Apply distortion correction
undistorted_img = cv2.undistort(img, intrinsic_matrix, distortion_coefficients)

# Show original and undistorted images
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done!")
