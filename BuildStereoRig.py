import sys
import os

import cv2

# C:\Users\14424\AppData\Local\Programs\Python\Python311\Lib\site-packages\simplestereo
import simplestereo as ss

"""
Build a stereo rig object calculating parameters from calibration images.
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))

# Set image folder(s)
# => Two ~120^ World Cams
#loadPath = os.path.join(curPath, "revok", "center")

# => Left World Cam + Right RGB Cam
img_dir_L = "X:\\data\\chessboard_e\\extrinsics_sync\\left\\"
img_dir_R = "X:\\data\\chessboard_e\\extrinsics_sync\\rgb\\"

# Destination
saveFile = os.path.join(curPath, "revok", "rig.json")

# Total number of images
N_IMAGES = 21
#N_IMAGES = 29

# Image paths
# NOTE: LEFT & RIGHT are JPEG, CENTER is .pgm
#images = [(os.path.join(loadPath, f'left ({i+1}).pgm'), os.path.join(loadPath, f'center ({i+1}).pgm')) for i in range(N_IMAGES)]
images = [(os.path.join(img_dir_L, f'left ({i+1}).jpg'), os.path.join(img_dir_R, f'rgb ({i+1}).jpg')) for i in range(N_IMAGES)]
#print(f"Calibrating using {len(images)} images from:\n{img_dir_L}+{img_dir_R}...")

# Calibrate and build StereoRig object
# Chessboard: (7,6) 52mm squares
# Don't calibrate right camera in rig: we set distortion coefficients to zero
rig = ss.calibration.chessboardHybridStereo(images, chessboardSize=(7,6), squareSize=52.0,
distortionCoeffsNumber=0)

# Save rig object to file
rig.save(saveFile)

# Print some info
print("Saved in:", saveFile)
print("Reprojection error:", rig.reprojectionError)
print("Centers:", rig.getCenters())
print("Baseline:", rig.getBaseline())

print("Done!")
