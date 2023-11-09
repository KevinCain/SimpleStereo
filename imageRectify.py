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

#imgPath = os.path.join(curPath,"revok","center")
img_dir_L = "X:\\data\\chessboard_e\\extrinsics_sync\\left\\"
img_dir_R = "X:\\data\\chessboard_e\\extrinsics_sync\\rgb\\"
#imgPath = "X:\\data\\chessboard_e\\subjects\\"

# StereoRig file
loadFile = os.path.join(curPath,"revok","rigRect.json")

# Load stereo rig from file
rigRect = ss.RectifiedStereoRig.fromFile(loadFile)

# Read right and left image (please ensure the order!!!)
#img1 = cv2.imread(os.path.join(imgPath,'left (1).pgm'))
#img2 = cv2.imread(os.path.join(imgPath,'center (1).pgm'))
img1 = cv2.imread(os.path.join(img_dir_L,'left (1).jpg'))
img2 = cv2.imread(os.path.join(img_dir_R,'rgb (1).jpg'))
#img1 = cv2.imread(os.path.join(imgPath,'ad_L.pgm'))
#img2 = cv2.imread(os.path.join(imgPath,'ad_C.pgm'))

# Optional -- alpha=0 may help removing unwanted border
rigRect.computeRectificationMaps(alpha=0)

# Simply rectify two images (it takes care of distortion too)
img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)

# Check if both rectified images have the same size
if img1_rect.shape != img2_rect.shape:
    raise ValueError("The rectified images have different sizes. Aborting.")

# Save the rectified images to disk
cv2.imwrite(os.path.join('im0.png'), img1_rect)
cv2.imwrite(os.path.join('im1.png'), img2_rect)

# Show images together
visImg = np.hstack((img1_rect, img2_rect))

# Draw some horizontal lines as reference
# (after rectification all horizontal lines are epipolar lines)
# Define colors for lines
color1 = (0, 10, 255)  # Red
color2 = (30, 255, 20)  # Green

# Define line thickness
thickness = 2

# Draw some horizontal lines as reference
# (after rectification all horizontal lines are epipolar lines)
for i, y in enumerate(range(50, visImg.shape[0], 50)):  # Draw a line every 50 pixels
    if i % 2 == 0:
        color = color1
    else:
        color = color2
    cv2.line(visImg, (0, y), (visImg.shape[1], y), color=color, thickness=thickness)

cv2.imshow('Rectified images', visImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
