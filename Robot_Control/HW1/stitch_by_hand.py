#!/usr/bin/env python3

from stitch import *


def main():
    img1_name = f'{photos_folder_path}/undistorted_photo1.png'
    img2_name = f'{photos_folder_path}/undistorted_photo2.png'

    matches = [((29, 252), (193, 260)), ((221, 503), (369, 501)), ((207, 133), (355, 139)), ((340, 203), (484, 205)),
               ((439, 479), (589, 486))]

    h_matrix = find_homography(matches)

    img1 = cv2.imread(img1_name)
    img2 = cv2.imread(img2_name)

    stitched_photo = stitch(img1, img2, h_matrix)
    show_photo(stitched_photo / 255)
    cv2.imwrite(f"{photos_folder_path}/hand_stitched_photo.png", stitched_photo)


if __name__ == "__main__":
    main()
