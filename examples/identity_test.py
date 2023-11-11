import numpy as np
import cv2
import glob
import os
import logging

def load_calibration_data(filename):
    with np.load(filename) as data:
        mtxL = data['mtxL']
        distL = data['distL']
        mtxR = data['mtxR']
        distR = data['distR']
    return mtxL, distL, mtxR, distR

def display_side_by_side(img1, img2, max_width=1800):
    # Combine images side by side
    combined_img = np.hstack((img1, img2))
    h, w = combined_img.shape[:2]
    
    # Draw grid lines
    spacing = 50  # pixels
    color = (0, 255, 0)  # Green
    thickness = 1

    for x in range(0, w, spacing):
        cv2.line(combined_img, (x, 0), (x, h), color, thickness)
        
    for y in range(0, h, spacing):
        cv2.line(combined_img, (0, y), (w, y), color, thickness)
        
    # Scale the image to fit within the max_width
    scale = max_width / float(w)
    if scale < 1.0:
        combined_img = cv2.resize(combined_img, (int(w * scale), int(h * scale)))

    # Display the image
    cv2.imshow('Original vs. Undistorted with Grid', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Load calibration data
    logging.info("Loading calibration data.")
    calib_file = 'calibration_data.npz'
    mtxL, distL, mtxR, distR = load_calibration_data(calib_file)
    logging.info(f"Loaded calibration data: mtxR={mtxR}, distR={distR}")

    # Image directories
    img_dir_R = "X:\\data\\chessboard_e\\rgb"

    # First image from the right camera sequence
    first_img_R = sorted(glob.glob(os.path.join(img_dir_R, '*.jpg')))[0]

    # Processing first image for Right camera
    logging.info("Processing first image for Right camera.")

    imgR = cv2.imread(first_img_R)
    h, w = imgR.shape[:2]

    # Using Identity Matrix for testing
    id_mtxR = np.identity(3)
    zero_distR = np.zeros((1, 5))

    new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (w, h), 1, (w, h))

    dstR = cv2.undistort(imgR, mtxR, distR, None, new_mtxR)
    id_dstR = cv2.undistort(imgR, id_mtxR, zero_distR, None, id_mtxR)

    # Display the original, calibrated undistorted, and identity matrix undistorted images side by side
    display_side_by_side(imgR, dstR, max_width=1800)
    #display_side_by_side(imgR, id_dstR, max_width=1800)
    #display_side_by_side(dstR, id_dstR, max_width=1800)
