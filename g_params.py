# capture raw image
raw_path = ".\\raw"
chess_champ_path = ".\\chess_champ_raw"

raw_top = 45
raw_bottom = 1037
raw_left = 540
raw_right = 1453
resolution = (1920, 1080)
# need to scale to 902x1007

# generate data
pre_generate_path = ".\\chess_champ_pre_generate"
train_image_path = ".\\train_image"

# save model
model_path = ".\\model"

# recognize chess champ
image_width = 913
image_height = 1017
image_train_size = 120
webcam_path = ".\\webcam"

# utils
chess_piece_size_image = 101.5
margin_image = 49.5

chess_table = [
    "b_gen_", "b_adv_", "b_ele_", "b_hor_", "b_cha_", "b_can_", "b_sol_",
    "r_gen_", "r_adv_", "r_ele_", "r_hor_", "r_cha_", "r_can_", "r_sol_",
]

chess_cn = [
    "一",
    "将", "士", "象", "馬", "車", "包", "兵",
    "帅", "仕", "相", "傌", "俥", "砲", "卒"
]

chess_eng = [
    "Black General", "Black Advisor", "Black Elephant", "Black Horse", "Black Chariot", "Black Cannon", "Black Soldier",
    "Red General", "Red Advisor", "Red Elephant", "Red Horse", "Red Chariot", "Red Cannon", "Red Soldier"
]

chess_vn = [
    "Tướng đen", "Sĩ đen", "Tượng đen", "Mã đen", "Xe đen", "Pháo đen", "Tốt đen",
    "Tướng đỏ", "Sĩ đỏ", "Tượng đỏ", "Mã đỏ", "Xe đỏ", "Pháo đỏ", "Tốt đỏ"
]

# train
chess_info = ".\\chess_info.txt"

# recognize
margin_real = 20
chess_piece_size_real = 41
height_real = 410
width_real = 368
