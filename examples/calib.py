import os
import cv2
import numpy as np
import glob
import logging

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        reprojected_imgpoints, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        reprojected_imgpoints = reprojected_imgpoints.reshape(imgpoints[i].shape)  # Match the shape
        error = cv2.norm(imgpoints[i], reprojected_imgpoints, cv2.NORM_L2) / len(reprojected_imgpoints)
        total_error += error
        total_points += 1

    mean_error = total_error / total_points if total_points else 0
    return mean_error

def calibrate_camera(img_dir, objp, board_size):
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    successful_imgs = 0
    
    pgm_files = glob.glob(os.path.join(img_dir, '*.pgm'))
    jpg_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    filenames = pgm_files + jpg_files
    logging.info(f"Loading .pgm and .jpg files from {img_dir}.")

    # Termination criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, fname in enumerate(filenames):
        logging.info(f"Processing image {i + 1}/{len(filenames)}")
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        if img is None:
            logging.warning(f"Failed to load image {fname}")
            continue

        ret, corners = cv2.findChessboardCorners(img, board_size, None)
        logging.info(f"findChessboardCorners returned {ret} for {fname}")

        if ret:
            successful_imgs += 1
            objpoints.append(objp)
            cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            '''
            # Visualize the corners
            img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_vis, board_size, corners, ret)
            cv2.imshow('Chessboard Corners', img_vis)
            cv2.waitKey(0)
            '''

    if successful_imgs == 0:
        logging.error("No corners found in any image.")
        return None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    logging.info(f"Successfully found corners in {successful_imgs} images.")

    # Logging reprojection error for each image
    for i in range(len(objpoints)):
        img_error = calculate_reprojection_error([objpoints[i]], [imgpoints[i]], [rvecs[i]], [tvecs[i]], mtx, dist)
        logging.info(f"Reprojection error for image {i + 1}: {img_error}")

    mean_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    logging.info(f"Total mean reprojection error: {mean_error:.6f}")

    return ret, mtx, dist, rvecs, tvecs, filenames

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    board_size = (7, 6)
    square_size = 1.0  # Define the square size according to your calibration board
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    logging.info("Calibrating left camera.")
    # 141 photos
    img_dir_L = "X:\\data\\chessboard_e\\left"
    # 21 photos
    #img_dir_L = "X:\\data\\chessboard_e\\extrinsics_sync\\left\\"
    img_dir_R = "X:\\data\\chessboard_e\\extrinsics_sync\\rgb\\"
    retL, mtxL, distL, rvecsL, tvecsL, filenames_L = calibrate_camera(img_dir_L, objp, board_size)
    
    if retL:
        np.savez('calibration_data_left_140.npz', mtxL=mtxL, distL=distL, rvecsL=rvecsL, tvecsL=tvecsL)
        #np.savez('calibration_data_left_21.npz', mtxL=mtxL, distL=distL, rvecsL=rvecsL, tvecsL=tvecsL)
        
        '''
        # Load the saved calibration data
        with np.load('calibration_data_left_21.npz') as X:
            mtxL, distL, rvecsL, tvecsL = [X[i] for i in ('mtxL', 'distL', 'rvecsL', 'tvecsL')]
        
        # Reproject the points and visualize on the images
        for idx, fname in enumerate(filenames_L):
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.warning(f"Failed to load image {fname}, skipping reprojection.")
                continue
            
            # Reproject the points
            reprojected_imgpoints, _ = cv2.projectPoints(objp, rvecsL[idx], tvecsL[idx], mtxL, distL)
            
            # Draw and display the reprojected points
            img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            imgpoints_vis = reprojected_imgpoints.reshape(-1, 2)  # Ensure the correct shape
            cv2.drawChessboardCorners(img_vis, board_size, imgpoints_vis, True)
            
            cv2.imshow('Reprojected Corners', img_vis)
            key = cv2.waitKey(0)
            if key == 27:  # Exit if ESC is pressed
                break
                
        cv2.destroyAllWindows()
        '''
    else:
        logging.error("Left camera calibration failed.")
        
    '''
    logging.info("Calibrating right camera.")
    retR, mtxR, distR, rvecsR, tvecsR, filenames_R = calibrate_camera(img_dir_R, objp, board_size)

    if retR:    
        np.savez('calibration_data_right_21.npz', mtxR=mtxR, distR=distR, rvecsR=rvecsR, tvecsR=tvecsR)
        logging.info("Calibration complete.")
    else:
        logging.error("Right camera calibration failed.")
    '''
