import numpy as np
from PIL import Image
import os


data_path = '../Dataset/LLCM/'
def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [data_path + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
file_path_train = data_path + 'idx/train_vis.txt'
file_path_val = data_path + 'idx/train_nir.txt'

files_rgb, id_train = load_data(file_path_train)
files_ir, id_val = load_data(file_path_val)

pid_container = set()
for label in id_val:
    pid = label
    pid_container.add(pid)

pid2label = {pid: label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288

def read_imgs(train_image, train_label):
    train_img = []
    train_lbl = []
    for index, (image, label) in enumerate(zip(train_image, train_label)):
        img = Image.open(image)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)
        pid = label
        pid = pid2label[pid]
        train_lbl.append(pid)
    return np.array(train_img), np.array(train_lbl)


# rgb imges
train_img, train_label = read_imgs(files_rgb, id_train)
np.save(data_path + 'train_rgb_resized_img.npy', train_img)
np.save(data_path + 'train_rgb_resized_label.npy', train_label)

# ir imges
train_img, train_label = read_imgs(files_ir, id_val)
np.save(data_path + 'train_ir_resized_img.npy', train_img)
np.save(data_path + 'train_ir_resized_label.npy', train_label)
