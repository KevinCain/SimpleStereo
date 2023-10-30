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
# Image folder
#loadPath = os.path.join(curPath, "revok", "chessboard_d")
loadPath = os.path.join(curPath, "revok", "center")
# Destination
saveFile = os.path.join(curPath, "revok", "rig.json")

# Total number of images
N_IMAGES = 11
#N_IMAGES = 29

# Image paths
# NOTE: LEFT are PGM, RIGHT are JPEG
#images = [(os.path.join(loadPath, f'leftb ({i+1}).pgm'), os.path.join(loadPath, f'rgbb ({i+1}).jpg')) for i in range(N_IMAGES)]
images = [(os.path.join(loadPath, f'left ({i+1}).pgm'), os.path.join(loadPath, f'center ({i+1}).pgm')) for i in range(N_IMAGES)]


print(f"Calibrating using {len(images)} images from:\n{loadPath}...")

# Calibrate and build StereoRig object
# Chessboard: (7,6) 52mm squares
# Use OpenCV fisheye calibration: https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
rig = ss.calibration.chessboardStereo(images, chessboardSize=(7,6), squareSize=52.0, fisheye1=0, fisheye2=0)

# Save rig object to file
rig.save(saveFile)

# Print some info
print("Saved in:", saveFile)
print("Reprojection error:", rig.reprojectionError)
print("Centers:", rig.getCenters())
print("Baseline:", rig.getBaseline())

print("Done!")
