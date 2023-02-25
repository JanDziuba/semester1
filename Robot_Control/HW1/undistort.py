#!/usr/bin/env python3

import cv2
from os import listdir
import numpy as np

photos_folder_path = './images'


def main():
    map1 = np.load('camera_calibration_data/map1.npy')
    map2 = np.load('camera_calibration_data/map2.npy')

    photo_names = [f for f in listdir(photos_folder_path) if f.endswith('.png')]

    for photo_name in photo_names:
        photo = cv2.imread(f'{photos_folder_path}/{photo_name}')
        undistorted_photo = cv2.remap(photo, map1, map2, cv2.INTER_LINEAR)
        cv2.imwrite(f"./images/undistorted_{photo_name}", undistorted_photo)


if __name__ == "__main__":
    main()
