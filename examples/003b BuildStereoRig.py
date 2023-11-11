import sys
import os

import cv2

import simplestereo as ss
"""
Build a stereo rig object calculating parameters from calibration images.
"""

# Total number of images
#N_IMAGES = 23
N_IMAGES = 22

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
#loadPath = os.path.join(curPath,"res","1","calib")    # Image folder
saveFile = os.path.join(curPath, "revok", "rig.json")

# Image paths
'''
img_dir_L = "X:\\data\\chessboard_e\\extrinsics_sync\\left\\"
img_dir_R = "X:\\data\\chessboard_e\\extrinsics_sync\\center\\"
images = [(os.path.join(img_dir_L, f'left ({i+1}).jpg'), os.path.join(img_dir_R, f'center ({i+1}).jpg')) for i in range(N_IMAGES)]
'''

img_dir_L = "X:\\data\\chessboard_e\\center_sync\\center\\"
img_dir_R = "X:\\data\\chessboard_e\\center_sync\\rgb\\"
images = [(os.path.join(img_dir_L, f'center ({i+1}).jpg'), os.path.join(img_dir_R, f'rgb ({i+1}).jpg')) for i in range(N_IMAGES)]

print(f"Calibrating using {len(images)} images from:\n{img_dir_L}...\n{img_dir_R}...")

# Calibrate and build StereoRig object
rig = ss.calibration.chessboardStereo( images, chessboardSize=(7,6), squareSize=60.5)

# Save rig object to file
rig.save( saveFile )

# Print some info
print("Saved in:", saveFile)
print("Reprojection error:", rig.reprojectionError)
print("Centers:", rig.getCenters())
print("Baseline:", rig.getBaseline())

print("Distortion coefficients 1:", rig.distCoeffs1)
print("Distortion coefficients 2:", rig.distCoeffs2)

print("Done!")
