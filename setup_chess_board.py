import cv2
import g_params
import recognize_chess_champ as rcc

# def distance_real(x):
#     margin_real = g_params.margin_real
#     cube_size = g_params.cube_height
#     cube_real = g_params.chess_piece_size_real
#     return round((x - margin_real)/cube_size)*cube_real + margin_real

image_name = "0.jpg"


def setup_chess_board(chess_x_real, chess_y_real, chess_int, show_result):
    f = open("setup_chess_board.txt", "w")
    if show_result:
        print(" ")
        print("Các tọa độ cần đến theo thứ tự: ")
    for i in range(32):
        if chess_int[i] == 4:  # b cha
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 3:  # b hor
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 2:  # b ele
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 1:  # b cha
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 0:  # b gen
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 11:  # r cha
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 10:  # r hor
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 9:  # r ele
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 8:  # r adv
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 7:  # r gen
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 12:  # r can
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 5:  # b can
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 6:  # b sol
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))

    for i in range(32):
        if chess_int[i] == 13:  # r sol
            f.write(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])) + "\n")
            if show_result:
                print(str(int(chess_x_real[i])) + "," + str(int(chess_y_real[i])))
    f.close()


def main():
    # locate chess champs' position and determine their type
    image = cv2.imread(g_params.webcam_path + "\\" + image_name, 0)
    chess_x, chess_y, chess_int = rcc.locate_chess_champ(image)
    chess_x_real, chess_y_real, chess_int = rcc.chess_board_generator(chess_x, chess_y, chess_int, show_real_location=False)
    # generate steps for setting up chess board -> send to plc
    if len(chess_int) != 32:
        print("Số lượng quân cờ không đúng hoặc phần mềm nhận diện sai!")
    else:
        setup_chess_board(chess_x_real, chess_y_real, chess_int, show_result=True)


if __name__ == '__main__':
    main()
