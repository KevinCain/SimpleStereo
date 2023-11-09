import os
import cv2
import shutil
import numpy as np

# Calculate the variance of the Laplacian to determine if the image is blurry
def is_blurry(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Resize the image to a given width while maintaining aspect ratio
def resize_image(image, width=2000):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    height = int(width / aspect_ratio)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# The path to the image directory
input_path = "X:\\data\\charcoa_a\\"
culled_folder = os.path.join(input_path, 'culled')

# Create 'culled' folder if it does not exist
if not os.path.exists(culled_folder):
    os.mkdir(culled_folder)

# List all files with prefix 'rgb'
#image_files = [f for f in os.listdir(input_path) if f.startswith('rgb') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files = [f for f in os.listdir(input_path) if f.startswith('rgb') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files = [os.path.join(input_path, f) for f in image_files]

# Loop through the list of image files
for img_path in image_files:
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get the blurriness score
    blurriness_score = is_blurry(gray)
    print(f"Blurriness score for {img_path}: {blurriness_score}")

    # Check if the image is blurry based on the blurriness score
    if blurriness_score < 40:  # 100 is just a sample threshold, you can adjust it
        # Resize the image to a width of 2000 while maintaining the aspect ratio
        resized_img = resize_image(img, width=2000)

        # Show the image
        cv2.imshow('Blurry Image', resized_img)
        
        # Wait for user input
        key = cv2.waitKey(0) & 0xFF
        
        # Check if the user pressed 'x'
        if key == ord('x'):
            # Move the image to the 'culled' folder
            shutil.move(img_path, os.path.join(culled_folder, os.path.basename(img_path)))
        
        # Destroy the image window
        cv2.destroyWindow('Blurry Image')
