"""
    Xac dinh ten cac quan co de lam bao cao.
    Mod tu recognize_chess_champ
"""

import cv2
import numpy as np
from keras.models import load_model
from keras.utils.image_utils import img_to_array
import g_params
import utils


show_cropped_image = False
thresh = 170
norm_size = 120
model_name = "model_moi.h5"
image_name = "c1-6.jpg"

left = g_params.raw_left
right = g_params.raw_right
top = g_params.raw_top
bottom = g_params.raw_bottom


def locate_chess_champ():
    # load model
    model = load_model(g_params.model_path + "\\" + model_name)
    print("model loaded!")

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
    data = []
    chess_int = []
    chess_x = []
    chess_y = []
    for circle in circles[0, :]:
        (x, y, _) = circle
        center = (x, y)
        chess_x.append(x)
        chess_y.append(y)
        cv2.circle(image, center, radius, (0, 0, 0), 2)
        count_circle = count_circle + 1

        # create data
        mask = np.zeros((width, height), np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
        masked_im = cv2.bitwise_and(image, image, mask=mask)
        cropped_im = masked_im[y - radius:y + radius, x - radius:x + radius]
        chess_champ = cv2.resize(cropped_im, (norm_size, norm_size))
        binary_chess_champ = utils.into_black_white(chess_champ, thresh=thresh,  show=show_cropped_image)
        data.append(img_to_array(binary_chess_champ))
    data = np.array(data)
    data = data/255.0
    # print("board data shape      = ", data.shape, " ===============================")
    for i in range(len(data)):
        img = data[i]
        img = np.expand_dims(img, 0)
        output = model.predict(img)
        chess_int.append(output.argmax())
    # print("chess_int.size = ", len(chess_int))
    del model
    return chess_x, chess_y, chess_int


def chess_board_generator(chess_x, chess_y, chess_int):
    size = len(chess_int)
    cube_height = g_params.chess_piece_size_image
    cube_width = g_params.chess_piece_size_real

    # init board cn
    board_cn = []
    for i in range(9):
        row = []
        for j in range(10):
            row.append("一")
        board_cn.append(row)
    print(" ")

    chess_cn = g_params.chess_cn
    chess_vn = g_params.chess_vn
    for i in range(size):
        x = round((chess_x[i] - g_params.margin_x) / cube_width)
        y = round((chess_y[i] - g_params.margin_y) / cube_height)
        board_cn[y][x] = chess_cn[chess_int[i] + 1]
        x_real = g_params.margin_real + g_params.chess_piece_size_real * x
        y_real = g_params.margin_real + g_params.chess_piece_size_real * y
        print(chess_vn[chess_int[i]] + " (" + str(y_real) + ", " + str(x_real) + ")")

    print(" ")
    print("Giá trị phân ngưỡng: ", thresh)
    print("Phát hiện được " + str(size) + " quân cờ trong ảnh.")
    print("Mô phỏng lại bàn cờ: ")
    for i in range(9):
        print(board_cn[i])


def main():
    # locate chess champs' position and determine their type
    chess_x, chess_y, chess_int = locate_chess_champ()

    # print chess board
    chess_board_generator(chess_x, chess_y, chess_int)


if __name__ == '__main__':
    main()
