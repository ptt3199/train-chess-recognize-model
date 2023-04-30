# locate chess champs' position and determine their type
import cv2
import numpy as np
from keras.models import load_model
from keras.utils.image_utils import img_to_array

# homemade library
import g_params
import utils
import calibrate

# set up
show_cropped_image = False
generate_map = False
train_size = g_params.image_train_size
model_name = "model_moi.h5"


left = g_params.raw_left
right = g_params.raw_right
top = g_params.raw_top
bottom = g_params.raw_bottom


def locate_chess_champ(image):
    # load model
    model = load_model(g_params.model_path + "\\" + model_name)
    print("model loaded!")

    width, height = image.shape
    # width = g_params.image_width
    # height = g_params.image_height
    # image = cv2.resize(image, (height, width))

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
    chess_int = []   # index to search chess name in g_params
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
        chess_champ = cv2.resize(cropped_im, (train_size, train_size))
        binary_chess_champ = utils.into_black_white(chess_champ, thresh=130, show=show_cropped_image)
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


def fix_real(x, y):
    # x, y = x_inp[:], y_inp[:]
    size = len(x)
    # x0, y0 = np.ones(size)*184, np.ones(size)*204
    x0, y0 = 184, 204
    r = np.sqrt(np.multiply(x - x0, x - x0) + np.multiply(y - y0, y - y0))
    dx = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]
    dy = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0]
    m = [60, 80, 110, 130, 140, 150, 160, 170, 180, 200, 230, 260]
    for i in range(size):
        # inc x, inc y
        if (m[0] < r[i] <= m[1]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[0]
            y[i] += dy[0]
        elif (m[1] < r[i] <= m[2]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[1]
            y[i] += dy[1]
        elif (m[2] < r[i] <= m[3]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[2]
            y[i] += dy[2]
        elif (m[3] < r[i] <= m[4]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[3]
            y[i] += dy[3]
        elif (m[4] < r[i] <= m[5]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[4]
            y[i] += dy[4]
        elif (m[5] < r[i] <= m[6]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[5]
            y[i] += dy[5]
        elif (m[6] < r[i] <= m[7]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[6]
            y[i] += dy[6]
        elif (m[7] < r[i] <= m[8]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[7]
            y[i] += dy[7]
        elif (m[8] < r[i] <= m[9]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[8]
            y[i] += dy[8]
        elif (m[9] < r[i] <= m[10]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[9]
            y[i] += dy[9]
        elif (m[10] < r[i] <= m[11]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[10]
            y[i] += dy[10]
        elif (m[11] < r[i]) and (x[i] < x0) and (y[i] < y0):
            x[i] += dx[11]
            y[i] += dy[11]
        # inc x, dec y
        elif (m[0] < r[i] <= m[1]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[0]
            y[i] -= dy[0]
        elif (m[1] < r[i] <= m[2]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[1]
            y[i] -= dy[1]
        elif (m[2] < r[i] <= m[3]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[2]
            y[i] -= dy[2]
        elif (m[3] < r[i] <= m[4]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[3]
            y[i] -= dy[3]
        elif (m[4] < r[i] <= m[5]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[4]
            y[i] -= dy[4]
        elif (m[5] < r[i] <= m[6]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[5]
            y[i] -= dy[5]
        elif (m[6] < r[i] <= m[7]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[6]
            y[i] -= dy[6]
        elif (m[7] < r[i] <= m[8]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[7]
            y[i] -= dy[7]
        elif (m[8] < r[i] <= m[9]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[8]
            y[i] -= dy[8]
        elif (m[9] < r[i] <= m[10]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[9]
            y[i] -= dy[9]
        elif (m[10] < r[i] <= m[11]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[10]
            y[i] -= dy[10]
        elif (m[11] < r[i]) and (x[i] < x0) and (y[i] > y0):
            x[i] += dx[11]
            y[i] -= dy[11]
        # dec x, dec y
        elif (m[0] < r[i] <= m[1]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[0]
            y[i] -= dy[0]
        elif (m[1] < r[i] <= m[2]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[1]
            y[i] -= dy[1]
        elif (m[2] < r[i] <= m[3]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[2]
            y[i] -= dy[2]
        elif (m[3] < r[i] <= m[4]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[3]
            y[i] -= dy[3]
        elif (m[4] < r[i] <= m[5]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[4]
            y[i] -= dy[4]
        elif (m[5] < r[i] <= m[6]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[5]
            y[i] -= dy[5]
        elif (m[6] < r[i] <= m[7]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[6]
            y[i] -= dy[6]
        elif (m[7] < r[i] <= m[8]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[7]
            y[i] -= dy[7]
        elif (m[8] < r[i] <= m[9]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[8]
            y[i] -= dy[8]
        elif (m[9] < r[i] <= m[10]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[9]
            y[i] -= dy[9]
        elif (m[10] < r[i] <= m[11]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[10]
            y[i] -= dy[10]
        elif (m[11] < r[i]) and (x[i] > x0) and (y[i] > y0):
            x[i] -= dx[11]
            y[i] -= dy[11]
        # dec x, inc y
        elif (m[0] < r[i] <= m[1]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[0]
            y[i] += dy[0]
        elif (m[1] < r[i] <= m[2]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[1]
            y[i] += dy[1]
        elif (m[2] < r[i] <= m[3]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[2]
            y[i] += dy[2]
        elif (m[3] < r[i] <= m[4]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[3]
            y[i] += dy[3]
        elif (m[4] < r[i] <= m[5]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[4]
            y[i] += dy[4]
        elif (m[5] < r[i] <= m[6]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[5]
            y[i] += dy[5]
        elif (m[6] < r[i] <= m[7]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[6]
            y[i] += dy[6]
        elif (m[7] < r[i] <= m[8]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[7]
            y[i] += dy[7]
        elif (m[8] < r[i] <= m[9]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[8]
            y[i] += dy[8]
        elif (m[9] < r[i] <= m[10]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[9]
            y[i] += dy[9]
        elif (m[10] < r[i] <= m[11]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[10]
            y[i] += dy[10]
        elif (m[11] < r[i]) and (x[i] > x0) and (y[i] < y0):
            x[i] -= dx[11]
            y[i] += dy[11]

    x = np.rint(x)
    y = np.rint(y)
    return x, y, r


def define_chess_champ(chess_x, chess_y, chess_int, generate_map):
    size = len(chess_int)
    x_real = np.round(np.multiply(g_params.width_real / g_params.image_width, chess_x), 0)
    y_real = np.round(np.multiply(g_params.height_real / g_params.image_height, np.ones(size) * g_params.image_height - chess_y), 0)
    x_real_fix = x_real.copy()
    y_real_fix = y_real.copy()
    x_real_fix, y_real_fix, distance = fix_real(x_real_fix, y_real_fix)
    for i in range(size):
        # print(g_params.chess_vn[chess_int[i]] + " (" + str(x_real[i]) + "," + str(y_real[i]) + ") -> (" + str(x_real_fix[i]) + "," + str(y_real_fix[i]) + ") " + str(distance[i]))
        print(g_params.chess_vn[chess_int[i]] + " (" + str(x_real[i]) + "," + str(y_real[i]) + ") -> (" + str(
            x_real_fix[i]) + "," + str(y_real_fix[i]) + ") ")
    np.savetxt('x_real.txt', x_real_fix)
    np.savetxt('y_real.txt', y_real_fix)

    if generate_map:
        chess_piece_size_image = g_params.chess_piece_size_image
        margin_image = g_params.margin_image

        board_cn = []
        for i in range(10):
            row = []
            for j in range(9):
                row.append("一")
            board_cn.append(row)

        chess_cn = g_params.chess_cn
        for i in range(size):
            x = round((chess_x[i] - margin_image) / chess_piece_size_image)
            y = round((chess_y[i] - margin_image) / chess_piece_size_image)
            board_cn[y][x] = chess_cn[chess_int[i] + 1]
        for i in range(10):
            print(board_cn[i])

    return x_real_fix, y_real_fix, chess_int


def main():
    # load image
    # file = open('location_real.txt')
    # for i in range(1):
    #     image_name = 'test213' + str(i) + '.jpg'
    #     image = cv2.imread(g_params.webcam_path + '\\' + image_name, 0)
    #
    #     image = calibrate.calibrate_remap_image(image)
    #     image = image[top:bottom, left:right]
    #     # need to rescale
    #     image_width = int(g_params.image_width)
    #     image_height = int(g_params.image_height)
    #     image = cv2.resize(image, (image_width, image_height))
    #     cv2.imwrite('image.jpg', image)
    #     show_image = cv2.resize(image, (368, 411))
    #     cv2.imshow('Img', show_image)
    #     chess_x, chess_y, chess_int = locate_chess_champ(image)
    #     # print chess board
    #     chess_x_real, chess_y_real, chess_int = define_chess_champ(chess_x, chess_y, chess_int,
    #                                                                generate_map=generate_map)
    #     cv2.waitKey(0)

    image_name = 'test213' + '.jpg'
    image = cv2.imread(g_params.webcam_path + '\\' + image_name, 0)

    # capture live

    image = calibrate.calibrate_remap_image(image)
    image = image[top:bottom, left:right]
    # need to rescale to 905x1010 = 368x411
    image = cv2.resize(image, (g_params.image_width, g_params.image_height))
    show_image = cv2.resize(image, (368, 411))
    cv2.imshow('Img', show_image)
    chess_x, chess_y, chess_int = locate_chess_champ(image)
    # print chess board
    chess_x_real, chess_y_real, chess_int = define_chess_champ(chess_x, chess_y, chess_int, generate_map=generate_map)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
