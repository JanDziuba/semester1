from assignment_2_lib import take_a_photo, drive

import cv2
import numpy as np


def get_ball_mask(photo):
    cv_photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)
    cv_photo = cv_photo[0:420, :, :]
    hsv_image = cv2.cvtColor(cv_photo, cv2.COLOR_BGR2HSV)

    # positive red hue margin
    lower1 = np.array([0, 100, 50])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower1, upper1)

    # negative red hue margin
    lower2 = np.array([160, 100, 50])
    upper2 = np.array([189, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower2, upper2)

    return mask1 + mask2


def check_if_ball_visible(photo):
    mask = get_ball_mask(photo)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False
    else:
        return True


def get_ball_pixel_size(photo):
    mask = get_ball_mask(photo)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return np.average(np.array([w, h]))


def forward_distance(photo):
    ball_pixel_size = get_ball_pixel_size(photo)
    distance = 125 / ball_pixel_size
    steps = int(5000 / 2.1207376170053927 * (distance - 0.5))
    return steps


def get_ball_x_coord(photo):
    mask = get_ball_mask(photo)
    center_x = np.argmax(np.sum(mask, axis=0))
    return center_x


def turn_until_ball_is_visible(car, dir_right=True):
    photo = take_a_photo(car)
    is_ball_visible = check_if_ball_visible(photo)
    while not is_ball_visible:
        if dir_right:
            drive(car, True, -1)
            drive(car, False, 1)
        else:
            drive(car, True, 1)
            drive(car, False, -1)
        photo = take_a_photo(car)
        is_ball_visible = check_if_ball_visible(photo)


def drive_in_direction(car, x_coord, photo_width):
    if (photo_width / 2) * 0.9 < x_coord < (photo_width / 2) * 1.1:
        drive(car, True, 0)
    elif x_coord > photo_width / 2:
        drive(car, True, -1)
    else:
        drive(car, True, 1)


def find_a_ball_in_direction(car, dir_right):
    photo = take_a_photo(car)
    photo_width = photo.shape[1]

    turn_until_ball_is_visible(car, dir_right)
    photo = take_a_photo(car)

    dist = forward_distance(photo)
    dist_to_car_front = 350

    while dist > dist_to_car_front:
        x_coord = get_ball_x_coord(photo)
        drive_in_direction(car, x_coord, photo_width)
        photo = take_a_photo(car)
        dist = forward_distance(photo)


def find_a_ball(car):
    find_a_ball_in_direction(car, True)


def get_blue_cylinders_mask(photo):
    cv_photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)
    cv_photo = cv_photo[0:420, :, :]
    hsv_image = cv2.cvtColor(cv_photo, cv2.COLOR_BGR2HSV)

    # dark blue hue margin
    lower1 = np.array([115, 100, 50])
    upper1 = np.array([125, 255, 255])
    mask = cv2.inRange(hsv_image, lower1, upper1)
    return mask


def get_green_cylinders_mask(photo):
    cv_photo = cv2.cvtColor(photo, cv2.COLOR_RGBA2BGR)
    cv_photo = cv_photo[0:420, :, :]
    hsv_image = cv2.cvtColor(cv_photo, cv2.COLOR_BGR2HSV)

    # green hue margin
    lower1 = np.array([45, 100, 50])
    upper1 = np.array([75, 255, 255])
    mask = cv2.inRange(hsv_image, lower1, upper1)
    return mask


def get_x_coord_between_cylinders(contours):
    if not contours:
        return None
    if len(contours) == 1:
        x1, _, w1, _ = cv2.boundingRect(contours[0])
        return x1 + w1 / 2

    x1, _, w1, _ = cv2.boundingRect(contours[0])
    x2, _, w2, _ = cv2.boundingRect(contours[1])

    return (x1 + w1 / 2) / 2 + (x2 + w2 / 2) / 2


def get_max_cylinder_pixel_width(contours):
    if not contours:
        return None
    if len(contours) == 1:
        _, _, w1, _ = cv2.boundingRect(contours[0])
        return w1

    _, _, w1, _ = cv2.boundingRect(contours[0])
    _, _, w2, _ = cv2.boundingRect(contours[1])

    return max(w1, w2)


def find_cylinders_center_x_coord(green_contours, blue_contours):
    if len(green_contours) == 2:
        return get_x_coord_between_cylinders(green_contours)
    elif len(blue_contours) == 2:
        return get_x_coord_between_cylinders(blue_contours)
    elif len(green_contours) == 1 and len(blue_contours) == 1:
        green_x_coord = get_x_coord_between_cylinders(green_contours)
        blue_x_coord = get_x_coord_between_cylinders(blue_contours)
        return 3 * green_x_coord - 2 * blue_x_coord
    elif len(green_contours) == 1:
        return get_x_coord_between_cylinders(green_contours)
    elif len(blue_contours) == 1:
        return get_x_coord_between_cylinders(blue_contours)


def turn_to_x_coord(car, x_coord, photo_width):
    turn_right = (x_coord > photo_width/2)
    number_of_turns = int(8*(abs(x_coord-photo_width/2)/(photo_width/2)))
    for turn_n in range(number_of_turns):
        if turn_n % 2 == 0:
            if turn_right:
                drive(car, True, -1)
            else:
                drive(car, True, 1)
        else:
            if turn_right:
                drive(car, False, 1)
            else:
                drive(car, False, -1)


def get_cylinders_contours(photo):
    green_mask = get_green_cylinders_mask(photo)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    blue_mask = get_blue_cylinders_mask(photo)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return green_contours, blue_contours


def move_a_ball(car):
    photo = take_a_photo(car)
    photo_width = photo.shape[1]
    green_contours, blue_contours = get_cylinders_contours(photo)

    # turn until the ball is visible
    cylinders_x_coord = find_cylinders_center_x_coord(green_contours, blue_contours)
    dir_right = (cylinders_x_coord > photo_width/2)
    turn_until_ball_is_visible(car, dir_right)

    photo = take_a_photo(car)
    green_contours, blue_contours = get_cylinders_contours(photo)
    cylinders_x_coord = find_cylinders_center_x_coord(green_contours, blue_contours)

    # turn so that car faces region behind the ball
    ball_x_coord = get_ball_x_coord(photo)
    turn_x_coord = ball_x_coord + 0.5*(ball_x_coord - cylinders_x_coord)
    turn_to_x_coord(car, turn_x_coord, photo_width)

    # drive behind the ball
    ball_dist = forward_distance(photo)
    for _ in range(ball_dist//400):
        drive(car, True, 0)

    dir_right = (turn_x_coord < photo_width/2)
    find_a_ball_in_direction(car, dir_right)

    while True:
        photo = take_a_photo(car)
        green_contours, blue_contours = get_cylinders_contours(photo)

        if not green_contours and not blue_contours:
            return

        if green_contours:
            if get_max_cylinder_pixel_width(green_contours) > 50:  # green cylinder is close
                return

        x_coord = find_cylinders_center_x_coord(green_contours, blue_contours)
        drive_in_direction(car, x_coord, photo_width)

