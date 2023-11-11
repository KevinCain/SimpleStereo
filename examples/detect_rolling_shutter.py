import cv2
import numpy as np

def detect_rolling_shutter_artifacts(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found.")
        return

    # Detect edges using Canny detector
    edges = cv2.Canny(img, 50, 150)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if lines is not None:
        deviations = []
        for rho, theta in lines[:, 0]:
            # Vertical lines should have theta around 0 or 180 degrees
            deviation = min(abs(theta), abs(theta - np.pi))
            deviations.append(deviation)

        # Calculate average deviation from vertical
        avg_deviation = np.mean(deviations)
        if avg_deviation > 0.1:  # Threshold is an example, tweak based on your requirements
            print(f"Possible rolling shutter detected, average deviation: {avg_deviation}")

if __name__ == '__main__':
    #image_path = 'X:\\data\\chessboard_e\\rgb\\rgb (29).jpg'
    image_path = 'X:\\data\\chessboard_e\\left\\left (10).pgm'
    detect_rolling_shutter_artifacts(image_path)
