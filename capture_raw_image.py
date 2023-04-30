"""
updated: 3-3-2023
1. capture raw image -> crop
2. hough transform to find circles
3. crop circle -> black and white -> save
"""

import cv2
import os
from PIL import Image
import numpy as np


# homemade library
import g_params
import utils

frame_height = g_params.resolution[0]
frame_width = g_params.resolution[1]

# path
raw_path = g_params.raw_path
chess_champ_path = g_params.chess_champ_path

left = g_params.raw_left
right = g_params.raw_right
top = g_params.raw_top
bottom = g_params.raw_bottom

# show
show_circled_image = False
show_cropped_image = False

# circle size
norm_size = 120

start_index = 0  # for naming the raw image
count_all = 0  # for naming the crop circle image


def hough_circle(image, count_all1):  # capture all chess champ and save
    # load image
    width, height = image.shape

    # hough transform
    radius = round(0.288 * max(width, height) / 10)
    circle_min_distance = round(2 * radius)
    circles = cv2.HoughCircles(
        image,  # input image, greyscale
        cv2.HOUGH_GRADIENT,
        1.0,  # dp, the inverse ratio of resolution
        circle_min_distance,  # Minimum distance between detected centers
        param1=300,
        param2=15,
        minRadius=round(radius * .98),
        maxRadius=round(radius * 1.01)
    )
    # locate circles
    circles = np.uint16(np.around(circles))
    count_circle = 0
    for circle in circles[0, :]:
        (x, y, _) = circle
        center = (x, y)

        # info
        cv2.circle(image, center, radius, (0, 0, 0), 3)
        count_circle = count_circle + 1

        # crop circles
        mask = np.zeros((width, height), np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
        masked_im = cv2.bitwise_and(image, image, mask=mask)

        cropped_im = masked_im[y-radius:y+radius, x-radius:x+radius]
        chess_champ = cv2.resize(cropped_im, (norm_size, norm_size))
        binary_chess_champ = utils.into_black_white(chess_champ, show=show_cropped_image)
        count_all1 += 1
        cv2.imwrite(os.path.join(chess_champ_path, str(count_all1) + ".jpg"), binary_chess_champ)

    # print info
    if count_circle == 0:
        print("count_circle = 0, no circles found!")
        return [], [], []
    print("count_circle   = ", count_circle)
    if show_circled_image:
        cv2.imshow("Hough circle of image", image)
        # cv2.imwrite(".\\circles.jpg", image)
        cv2.waitKey(0)
    return count_all


def get_raw_image(start_index1):
    capture = cv2.VideoCapture(1)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    count = 0
    while True:
        # capture raw
        _, image = capture.read()
        image = image[top:bottom, left:right]

        # save crop
        name = str(count + start_index1) + ".jpg"
        cv2.imshow(name, image)  # imshow
        flag = cv2.waitKey(0)  # press enter to save image, and Esc to quit
        if flag == 13:
            save_path = os.path.join(raw_path, name)
            cv2.imwrite(save_path, image)
            if not cv2.imwrite(save_path, image):
                raise Exception("Could not write image")
            im_saved = Image.open(save_path)
            print(name, " \tsaved. pixel: ", str(im_saved.size[0]), "*", str(im_saved.size[1]))
            cv2.destroyWindow(name)
        count = count + 1
        if flag == 27:
            print("Quit: last image: " + name)
            cv2.destroyWindow(name)
            break
        if flag != 13 and flag != 27:
            cv2.destroyWindow(name)
            continue


def make_image_list():
    images = os.listdir(raw_path)
    count_image = 0
    image_list = []
    for img in images:
        image_full_path = os.path.join(raw_path, img)
        count_image += 1
        image_list.append(image_full_path)
    return image_list, count_image


def crop_chess_champ(image_list):
    for index in range(len(image_list)):
        raw_image = cv2.imread(image_list[index], 0)
        print(image_list[index])
        if raw_image is None:
            print("ERROR: circle.py line: 36, image loading failed!")
            return -888
        count_all1 = hough_circle(raw_image, count_all)
        print(count_all1)


def main():
    # get_raw_image(start_index)
    image_list, count_image = make_image_list()
    crop_chess_champ(image_list)


if __name__ == '__main__':
    main()
