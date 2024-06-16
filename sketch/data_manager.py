from __future__ import print_function, absolute_import
import os
import numpy as np
import random

def process_test_pku(img_dir, modal = 'sketch'):
    
    input_data_path = img_dir + 'trainPKUList_test' + '.txt'
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        if modal=='visible':
            file_image = [s.split(' ')[-1] for s in data_file_list if s.split(' ')[-2]=='2']
            file_label = [int(s.split(' ')[1])-1 for s in data_file_list if s.split(' ')[-2]=='2']
        elif modal=='sketch':
            file_image = [s.split(' ')[-1] for s in data_file_list if s.split(' ')[-2]=='1']
            file_label = [int(s.split(' ')[1])-1 for s in data_file_list if s.split(' ')[-2]=='1'] 
            
    file_image = [os.path.join(img_dir, f) for f in file_image]
    return file_image, np.array(file_label)