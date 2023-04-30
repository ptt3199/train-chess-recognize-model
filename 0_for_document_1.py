"""
    Xac dinh vi tri cac quan co de lam bao cao.
    Mod lai tu file recognize_chess_champ
"""

import cv2
import numpy as np
import g_params

show_cropped_image = False
norm_size = 120
image_name = "c1-6.jpg"

left = g_params.raw_left
right = g_params.raw_right
top = g_params.raw_top
bottom = g_params.raw_bottom


def locate_chess_champ():
    # load image
    image = cv2.imread(g_params.webcam_path + "\\" + image_name, 0)

    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # image = image[top:bottom, left:right]

    width = g_params.image_width
    height = g_params.image_height
    image = cv2.resize(image, (height, width))

    radius = round(0.288 * g_params.chess_piece_size_image)
    circle_min_distance = round(2 * radius)

    # hough transform
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
    # circles
    circles = np.uint16(np.around(circles))
    count_circle = 0
    chess_x = []
    chess_y = []
    for circle in circles[0, :]:
        (x, y, _) = circle
        center = (x, y)
        chess_x.append(x)
        chess_y.append(y)
        cv2.circle(image, center, radius, (0, 0, 0), 2)
    return chess_x, chess_y


def chess_board_generator(chess_x, chess_y):
    size = len(chess_x)
    cube_height = g_params.chess_piece_size_image
    cube_width = g_params.cub_width

    # init board
    board = []
    for i in range(9):
        row = []
        for j in range(10):
            row.append("__")
        board.append(row)
    margin_y = g_params.margin_y
    margin_x = g_params.margin_x
    for i in range(size):
        x = round((chess_x[i] - margin_x) / cube_width)
        y = round((chess_y[i] - margin_y) / cube_height)
        board[y][x] = "XX"
    for i in range(9):
        print(board[i])

    print("Phát hiện được " + str(size) + " quân cờ trong ảnh.")
    print("Tọa độ các quân cờ từ trái sang phải: ")
    for x in range(10):
        for y in range(9):
            if board[y][x] == "XX":
                x_real = g_params.margin_real + g_params.chess_piece_size_real * x
                y_real = g_params.margin_real + g_params.chess_piece_size_real * y
                print("(" + str(x_real) + ", " + str(y_real) + ")")

def main():
    # locate chess champs' position and determine their type
    chess_x, chess_y = locate_chess_champ()

    # print chess board
    chess_board_generator(chess_x, chess_y)


if __name__ == '__main__':
    main()
