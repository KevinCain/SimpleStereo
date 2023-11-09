import numpy as np
import cv2
import os

class StereoImageProcessor:
    # ...

    def loadRectificationMaps(self, load_path='C:\\temp\\rectification_maps'):
        fs_mapx1 = cv2.FileStorage(os.path.join(load_path, 'mapx1.yml'), cv2.FILE_STORAGE_READ)
        self.mapx1 = fs_mapx1.getNode('mapx1').mat()
        fs_mapx1.release()

        fs_mapy1 = cv2.FileStorage(os.path.join(load_path, 'mapy1.yml'), cv2.FILE_STORAGE_READ)
        self.mapy1 = fs_mapy1.getNode('mapy1').mat()
        fs_mapy1.release()

        fs_mapx2 = cv2.FileStorage(os.path.join(load_path, 'mapx2.yml'), cv2.FILE_STORAGE_READ)
        self.mapx2 = fs_mapx2.getNode('mapx2').mat()
        fs_mapx2.release()

        fs_mapy2 = cv2.FileStorage(os.path.join(load_path, 'mapy2.yml'), cv2.FILE_STORAGE_READ)
        self.mapy2 = fs_mapy2.getNode('mapy2').mat()
        fs_mapy2.release()

    def rectifyImages(self, img1, img2, interpolation=cv2.INTER_LINEAR):
        if self.mapx1 is None or self.mapx2 is None:
            print("Error: Rectification maps are not loaded.")
            return None, None

        img1_rect = cv2.remap(img1, self.mapx1, self.mapy1, interpolation)
        img2_rect = cv2.remap(img2, self.mapx2, self.mapy2, interpolation)
        return img1_rect, img2_rect

if __name__ == "__main__":
    # Paths
    curPath = os.path.dirname(os.path.realpath(__file__))
    imgPath = "X:\\data\\chessboard_e\\subjects\\"
    #img1_path = os.path.join(imgPath, 'ad_L.pgm')
    #img2_path = os.path.join(imgPath, 'ad_C.pgm')
    img1_path = os.path.join(imgPath, 'door_L.pgm')
    img2_path = os.path.join(imgPath, 'door_C.pgm')
    #img1_path = os.path.join(imgPath, 'bath_L.pgm')
    #img2_path = os.path.join(imgPath, 'bath_C.pgm')

    # Initialize StereoImageProcessor
    processor = StereoImageProcessor()

    # Load rectification maps
    processor.loadRectificationMaps(load_path='C:\\temp')

    # Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        print("Error: Unable to read the images.")
    else:
        # Rectify images
        img1_rect, img2_rect = processor.rectifyImages(img1, img2)

        # Check if rectification was successful
        if img1_rect is not None and img2_rect is not None:
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
        else:
            print("Rectification failed.")
