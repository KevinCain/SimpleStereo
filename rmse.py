import sys
import os

import numpy as np
import cv2
import random
#import simplestereo as ss

"""
Compute SIFT for feature matching and FLANN for fast approximate nearest neighbors searching. The RMSE is calculated based on the differences
in the x-coordinates of matching points, which should ideally be zero for perfectly rectified images.
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
# For sanity check images uncomment:
imgPath = os.path.join(curPath, "ref", "full_size")

#imgPath = os.path.join(curPath)

# Read right and left image
img1 = cv2.imread(os.path.join(imgPath,'im0.png'))
img2 = cv2.imread(os.path.join(imgPath,'im1.png'))

if img1 is None or img2 is None:
    print("Could not load images.")
    sys.exit(1)

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print("Matching in progress...")

def find_high_quality_matches(img1, img2):
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use a more accurate matcher - Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter out matches where y-delta is > 5 pixels
    good = []
    for match in matches:
        pt1 = np.array(kp1[match.queryIdx].pt)
        pt2 = np.array(kp2[match.trainIdx].pt)
        
        if abs(pt1[1] - pt2[1]) <= 5:
            good.append(match)

    return good, kp1, kp2

def find_matching_points(img1, img2):
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use FLANN based matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store good matches using Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good, kp1, kp2

def calculate_rmse(good, kp1, kp2, img_shape1, img_shape2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Calculate RMSE along x-axis (epipolar lines)
    diff = src_pts[:, 0, 0] - dst_pts[:, 0, 0]
    rmse = np.sqrt(np.mean(diff ** 2))

    # Scale by average resolution
    avg_width = (img_shape1[1] + img_shape2[1]) / 2
    rmse_scaled = rmse / avg_width
    return rmse_scaled

def calculate_mean_absolute_error(good, kp1, kp2, img_shape1, img_shape2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Calculate MAE along x-axis (epipolar lines)
    diff = src_pts[:, 0, 0] - dst_pts[:, 0, 0]
    mae = np.mean(np.abs(diff))

    # Scale by average resolution
    avg_width = (img_shape1[1] + img_shape2[1]) / 2
    mae_scaled = mae / avg_width
    return mae_scaled

if __name__ == "__main__":
    # Use FLANN
    #good, kp1, kp2 = find_matching_points(gray1, gray2)

    # Use brute force matching
    good, kp1, kp2 = find_high_quality_matches(gray1, gray2)

    # Randomly select 50 good matches to display
    selected_matches = random.sample(good, min(50, len(good)))

    # Draw the matches with circles
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, selected_matches, None, flags=2)

    for pt in selected_matches:
        cv2.circle(img1, tuple(map(int, kp1[pt.queryIdx].pt)), 8, (0, 255, 0), -1)
        cv2.circle(img2, tuple(map(int, kp2[pt.trainIdx].pt)), 8, (0, 0, 255), -1)

    # Resize image
    img_matches = cv2.resize(img_matches, (2200, int(2200 * img_matches.shape[0] / img_matches.shape[1])))

    cv2.imwrite(os.path.join(curPath, "ref", "matches.jpeg"), img_matches)

    # Display the matches
    cv2.imshow('Matching points', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate scaled RMSE
    rmse_scaled = calculate_rmse(good, kp1, kp2, gray1.shape, gray2.shape)
    print(f"Scaled RMSE along epipolar lines (x-axis): {rmse_scaled}")

    # Calculate scaled MAE
    mae_scaled = calculate_mean_absolute_error(good, kp1, kp2, gray1.shape, gray2.shape)
    print(f"Scaled MAE along epipolar lines (x-axis): {mae_scaled}")
