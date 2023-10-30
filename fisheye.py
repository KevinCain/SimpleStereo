import os
import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize variables for chessboard detection
num_corners_x = 7  # Number of internal corners along x-axis
num_corners_y = 6  # Number of internal corners along y-axis
board_size = (num_corners_x, num_corners_y)

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))

# Lists to store object points and image points from all the images
objpointsL, imgpointsL = [], []  # Left camera
objpointsR, imgpointsR = [], []  # Right camera

logging.info("Preparing object points.")
# Prepare object points
objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)

logging.info("Starting to process 29 stereo pairs.")
for i in range(1, 30):  # Loop through 29 stereo pairs
    logging.info(f"Processing pair {i}.")
    imgPathL = os.path.join(curPath, 'revok', 'center', f'left ({i}).pgm')
    imgPathR = os.path.join(curPath, 'revok', 'center', f'center ({i}).pgm')

    imgL = cv2.imread(imgPathL, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgPathR, cv2.IMREAD_GRAYSCALE)

    retL, cornersL = cv2.findChessboardCorners(imgL, board_size, None)
    retR, cornersR = cv2.findChessboardCorners(imgR, board_size, None)

    if retL and retR:
        logging.info(f"Found valid chessboard corners for pair {i}.")
        logging.info(f"   Left corners: {len(cornersL)}.")
        logging.info(f"   Right corners: {len(cornersR)}.")
        objpointsL.append(np.array(objp))
        imgpointsL.append(np.array(cornersL))
        objpointsR.append(np.array(objp))
        imgpointsR.append(np.array(cornersR))
    else:
        # If corners are invalid, show the images to the user
        if not retL or not retR:
            logging.warning(f"Invalid chessboard corners for pair {i}.")

        if cornersL is not None:
            cv2.drawChessboardCorners(imgL, board_size, cornersL, retL)
            logging.warning(f"Number of corners detected in left image: {len(cornersL)}")
        else:
            logging.warning("No corners detected in left image.")

        if cornersR is not None:
            cv2.drawChessboardCorners(imgR, board_size, cornersR, retR)
            logging.warning(f"Number of corners detected in right image: {len(cornersR)}")
        else:
            logging.warning("No corners detected in right image.")

        cv2.imshow(f'Invalid Left Image {i}', imgL)
        cv2.imshow(f'Invalid Right Image {i}', imgR)
        cv2.waitKey(0)

logging.info("Completed all tasks.")
cv2.destroyAllWindows()

logging.info(f"Object points data type: {objpointsL[0].dtype}, shape: {objpointsL[0].shape}")

if imgpointsL:
    logging.info(f"Image points left data type: {imgpointsL[0].dtype}, shape: {imgpointsL[0].shape}")

if imgpointsR:
    logging.info(f"Image points right data type: {imgpointsR[0].dtype}, shape: {imgpointsR[0].shape}")

logging.info(f"Length of objpointsL: {len(objpointsL)}, length of imgpointsL: {len(imgpointsL)}")
logging.info(f"Length of objpointsR: {len(objpointsR)}, length of imgpointsR: {len(imgpointsR)}")

objpointsL = [np.reshape(o, (-1, 1, 3)) for o in objpointsL]
imgpointsL = [np.reshape(i, (-1, 1, 2)) for i in imgpointsL]
objpointsR = [np.reshape(o, (-1, 1, 3)) for o in objpointsR]
imgpointsR = [np.reshape(i, (-1, 1, 2)) for i in imgpointsR]

objpointsL = [o.astype(np.float32) for o in objpointsL]
imgpointsL = [i.astype(np.float32) for i in imgpointsL]
objpointsR = [o.astype(np.float32) for o in objpointsR]
imgpointsR = [i.astype(np.float32) for i in imgpointsR]

logging.info(f"Sample object points: {objpointsL[0][:3]}")  # log first 3 object points
logging.info(f"Sample image points L: {imgpointsL[0][:3]}")  # log first 3 image points for left camera
logging.info(f"Sample image points R: {imgpointsR[0][:3]}")  # log first 3 image points for right camera


logging.info("Performing fisheye calibration.")
retL, mtxL, distL, rvecsL, tvecsL = cv2.fisheye.calibrate(
    objpointsL, imgpointsL, imgL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.fisheye.calibrate(
    objpointsR, imgpointsR, imgR.shape[::-1], None, None)



logging.info("Applying distortion correction.")
# Undistort images
for i in range(1, 30):
    logging.info(f"Applying distortion correction to pair {i}.")
    imgPathL = os.path.join(curPath, 'revok', 'center', f'left ({i}).pgm')
    imgPathR = os.path.join(curPath, 'revok', 'center', f'center ({i}).pgm')

    imgL = cv2.imread(imgPathL)
    imgR = cv2.imread(imgPathR)

    undistorted_imgL = cv2.fisheye.undistortImage(imgL, mtxL, distL)
    undistorted_imgR = cv2.fisheye.undistortImage(imgR, mtxR, distR)

    #cv2.imshow(f'Undistorted Left Image {i}', undistorted_imgL)
    #cv2.imshow(f'Undistorted Right Image {i}', undistorted_imgR)
    #cv2.waitKey(0)

logging.info(f"Calibration retL: {retL}, retR: {retR}")
logging.info(f"Calibration Matrix L: {mtxL}, R: {mtxR}")
logging.info(f"Distortion Coefficients L: {distL}, R: {distR}")

# New code to project object points back to source images
for i, objp in enumerate(objpointsL):
    # Read original images
    imgPathL = os.path.join(curPath, 'revok', 'center', f'left ({i + 1}).pgm')
    imgPathR = os.path.join(curPath, 'revok', 'center', f'center ({i + 1}).pgm')
    imgL = cv2.imread(imgPathL)
    imgR = cv2.imread(imgPathR)
    
    # For left camera
    reprojected_imgpointsL, _ = cv2.fisheye.projectPoints(objp, rvecsL[i], tvecsL[i], mtxL, distL)
    reprojected_imgpointsL = np.squeeze(reprojected_imgpointsL).astype(int)  # remove single-dimensional entries and cast to int

    # Draw reprojected points on the original image
    for pt in reprojected_imgpointsL:
        cv2.circle(imgL, tuple(pt), 3, (0, 0, 255), -1)  # draw a red circle
    
    # Show original image with reprojected points
    cv2.imshow(f'Reprojected Left Image {i + 1}', imgL)
    
    # For right camera
    reprojected_imgpointsR, _ = cv2.fisheye.projectPoints(objp, rvecsR[i], tvecsR[i], mtxR, distR)
    reprojected_imgpointsR = np.squeeze(reprojected_imgpointsR).astype(int)  # remove single-dimensional entries and cast to int

    # Draw reprojected points on the original image
    for pt in reprojected_imgpointsR:
        cv2.circle(imgR, tuple(pt), 3, (0, 0, 255), -1)  # draw a red circle
    
    # Show original image with reprojected points
    cv2.imshow(f'Reprojected Right Image {i + 1}', imgR)

    # Wait for user input
    cv2.waitKey(0)

logging.info("Completed all tasks.")
cv2.destroyAllWindows()
