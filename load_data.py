# -*- coding: utf-8 -*-

# import tarfile
import os
import tensorflow as tf
import keras
import platform
import g_params
import sys

sys.dont_write_bytecode = True
image_train_size = g_params.image_train_size

CHESS_TABLE = [
    "b_gen_",
    "b_adv_",
    "b_ele_",
    "b_hor_",
    "b_cha_",
    "b_can_",
    "b_sol_",
    "r_gen_",
    "r_adv_",
    "r_ele_",
    "r_hor_",
    "r_cha_",
    "r_can_",
    "r_sol_",
]

CHESS_CN = [
    "一",
    "将",
    "士",
    "象",
    "馬",
    "車",
    # "砲",
    "b",
    "兵",
    "帅",
    "仕",
    "相",
    "傌",
    "俥",
    # "炮",
    "r",
    "卒"
]


def str2int(string):
    return CHESS_TABLE.index(string)


def int2str(index):
    return CHESS_TABLE[index]


def int2cn(index):
    return CHESS_CN[index]


def document_chess_info(path_chess_choose):
    # environment
    print("tensorflow->", tf.__version__)
    print("keras     ->", keras.__version__)
    print("python    ->", platform.python_version())
    chess_info_txt = g_params.chess_info
    f = open(chess_info_txt, 'w')
    count = 0
    files = os.listdir(path_chess_choose)
    for filename in files:
        print_file_name = "Name -> " + str(count) + " -> " + filename
        print(print_file_name)
        count += 1
    index = 0
    all_chess_data_path = []
    for filename in files:
        temp_dir = path_chess_choose + "\\" + filename
        temp_files = os.listdir(temp_dir)
        for tempFilename in temp_files:
            all_chess_data_path.append(os.path.join(temp_dir, tempFilename))
            file_dir = str(index) + " " + filename + " " + tempFilename + "\n"
            f.write(file_dir)
        index += 1
    print("chess information documentation done!")
    return all_chess_data_path

#
# def main():
#     path_chess_choose = g_params.train_image_path
#     all_chess_data_path = document_chess_info(path_chess_choose)
#
#
# if __name__ == '__main__':
#     main()
