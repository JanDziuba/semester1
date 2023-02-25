#!/usr/bin/env python3

import cv2
import numpy as np

photos_folder_path = './images'


def show_photo(photo):
    cv2.imshow('photo', photo)
    cv2.waitKey(0)


def apply_homography(h_matrix, pixels_matrix):
    new_pixels = h_matrix @ pixels_matrix
    return np.rint((new_pixels / new_pixels[2])[:2]).astype(int)


# find positions of img corners after applying homography
def get_new_corners(img, h_matrix):
    height = img.shape[0]
    width = img.shape[1]

    p1 = np.array((0, 0, 1))
    p2 = np.array((height, 0, 1))
    p3 = np.array((height, width, 1))
    p4 = np.array((0, width, 1))

    original_corners = np.column_stack((p1, p2, p3, p4))
    return apply_homography(h_matrix, original_corners)


def find_dist_from_border(height, width, max_height, max_width):
    return min(max_height / 2 - abs(height - max_height / 2),
               max_width / 2 - abs(width - max_width / 2)) + 1


def stitch(img1, img2, h_matrix):
    new_img1_corners = get_new_corners(img1, h_matrix)

    img1_heights = new_img1_corners[0]
    img1_widths = new_img1_corners[1]

    max_img1_height = max(img1_heights)
    min_img1_height = min(img1_heights)
    max_img1_width = max(img1_widths)
    min_img1_width = min(img1_widths)

    max_height = max(max_img1_height, img2.shape[0])
    min_height = min(min_img1_height, 0)
    max_width = max(max_img1_width, img2.shape[1])
    min_width = min(min_img1_width, 0)

    stitched_img = np.zeros([max_height - min_height, max_width - min_width, 3])

    # fill stitched_img with pixels from img2
    for img2_h in range(img2.shape[0]):
        for img2_w in range(img2.shape[1]):
            stitched_img[img2_h - min_height][img2_w - min_width] = img2[img2_h][img2_w]

    inv_h_matrix = np.linalg.inv(h_matrix)

    transformed_pixels = np.array(np.meshgrid([np.arange(min_img1_height, max_img1_height)],
                                              [np.arange(min_img1_width, max_img1_width)], [1])).reshape(3, -1)

    old_pixels = apply_homography(inv_h_matrix, transformed_pixels)

    # fill stitched_img with pixels from img1, in overlapping area use weighted average
    for index in range(transformed_pixels.shape[1]):
        new_height = transformed_pixels[0][index]
        new_width = transformed_pixels[1][index]
        old_height = old_pixels[0][index]
        old_width = old_pixels[1][index]
        stitched_height = new_height - min_height
        stitched_width = new_width - min_width
        if 0 <= old_height < img1.shape[0] and 0 <= old_width < img1.shape[1]:
            if 0 <= new_height < img2.shape[0] and 0 <= new_width < img2.shape[1]:
                img1_weight = find_dist_from_border(old_height, old_width, img1.shape[0], img1.shape[1])
                img2_weight = find_dist_from_border(new_height, new_width, img2.shape[0], img2.shape[1])
                stitched_img[stitched_height][stitched_width] = np.average([img1[old_height][old_width],
                                                                            img2[new_height][new_width]],
                                                                           weights=[img1_weight, img2_weight],
                                                                           axis=0)
            else:
                stitched_img[stitched_height][stitched_width] = img1[old_height][old_width]

    return stitched_img


# matches - List[((w1, h1), (w2, h2))]
def find_homography(matches):
    A = []

    for (w1, h1), (w2, h2) in matches:
        A.append([h1, w1, 1, 0, 0, 0, -h2 * h1, -h2 * w1, -h2])
        A.append([0, 0, 0, h1, w1, 1, -w2 * h1, -w2 * w1, -w2])
    A = np.array(A)

    _, _, V = np.linalg.svd(A)

    h_matrix = np.reshape(V[-1], (3, 3))
    h_matrix = h_matrix / h_matrix[-1][-1]
    return h_matrix


# pair - ((w1, h1), (w2, h2))
def RANSAC_error(pair, h_matrix):
    p1 = np.array([pair[0][1], pair[0][0], 1])
    p2 = np.array([pair[1][1], pair[1][0], 1])

    p2_h = h_matrix @ p1.T
    p2_h = p2_h / p2_h[2]

    return np.linalg.norm(p2.T - p2_h)


# matches - List[((w1, h1), (w2, h2))]
def RANSAC(matches, num_iters=1000, max_err=200):
    best_inliers = 0
    best_h_matrix = None
    for i in range(num_iters):
        pairs = [matches[i] for i in np.random.choice(len(matches), 4)]

        h_matrix = find_homography(pairs)
        inliers = len([m for m in matches if RANSAC_error(m, h_matrix) < max_err])

        if inliers > best_inliers:
            best_inliers = inliers
            best_h_matrix = h_matrix

    return best_h_matrix


# This function is heavily inspired by an OpenCV tutorial:
# https://docs.opencv.org/4.6.0/dc/dc3/tutorial_py_matcher.html
#
# You can use it as is, you do not have to understand the insides.
# You need to pass filenames as arguments.
# You can disable the preview with visualize=False.
# lowe_ratio controls filtering of matches, increasing it
# will increase number of matches, at the cost of their quality.
#
# Return format is a list of matches, where match is a tuple of two keypoints.
# First keypoint designates coordinates on the first image.
# Second one designates the same feature on the second image.
def get_matches(filename1, filename2, visualize=True, lowe_ratio=0.6):
    # Read images from files, convert to greyscale
    img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < lowe_ratio * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    if visualize:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imshow("vis", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return good_matches


def main():
    img1_name = f'{photos_folder_path}/undistorted_photo1.png'
    img2_name = f'{photos_folder_path}/undistorted_photo2.png'

    matches = get_matches(img1_name, img2_name, visualize=False)

    h_matrix = RANSAC(matches)

    img1 = cv2.imread(img1_name)
    img2 = cv2.imread(img2_name)

    stitched_photo = stitch(img1, img2, h_matrix)
    show_photo(stitched_photo / 255)
    cv2.imwrite(f"{photos_folder_path}/stitched_photo.png", stitched_photo)


if __name__ == "__main__":
    main()
