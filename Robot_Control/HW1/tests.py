#!/usr/bin/env python3

from stitch import *


# tests with error from apply_homography finding the nearest int
def test_find_homography():
    id_m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    skew_m = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
    transl_m = np.array([[1, 0, 50], [0, 1, 100], [0, 0, 1]])
    rot_m = np.array([[0.7071, -0.7071, 0], [0.7071, 0.7071, 0], [0, 0, 1]])
    rot_skew_m = np.array([[0.7071, -0.7071, 0], [0.7071, 0.7071, 0], [0.001, 0, 1]])

    matrices = [id_m, skew_m, transl_m, rot_m, rot_skew_m]

    points_homo = np.array([[1, 1, 1], [0, 0, 1], [100, 100, 1], [-50, 69, 1], [3, 1000, 1]]).T

    for matrix in matrices:
        points_t = apply_homography(matrix, points_homo)
        points = points_homo[:2]
        matches = []
        for idx in range(points.shape[1]):
            matches.append(((points[1][idx], points[0][idx]), (points_t[1][idx], points_t[0][idx])))

        homography = find_homography(matches)

        print(matrix)
        print(homography)


def random_test_find_homography(num_iters=10, points_num=4):

    for _ in range(num_iters):
        h_matrix = np.random.rand(3, 3)
        h_matrix = h_matrix / h_matrix[-1][-1]

        points = np.random.rand(2, points_num)
        points = np.vstack((points, np.ones(points_num)))

        points_t = h_matrix @ points
        points_t = (points_t/points_t[2])[:2]
        points = points[:2]

        matches = []
        for idx in range(points.shape[1]):
            matches.append(((points[1][idx], points[0][idx]), (points_t[1][idx], points_t[0][idx])))

        homography = find_homography(matches)

        print(h_matrix)
        print(homography)


def main():
    random_test_find_homography()


if __name__ == "__main__":
    main()
