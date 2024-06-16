import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math

class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, gray = 2):
        self.gray = gray

    def __call__(self, img):
    
        idx = random.randint(0, self.gray)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img
        
        


class PKUdata(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, sketchIndex = None):
        # Load training images (path) and labels
        data_dir = '../autodl-tmp/PKU_sketch/'
        train_color_list   = data_dir + 'trainPKUList_train'+ '.txt'
        train_sketch_list = data_dir + 'trainPKUList_train'+ '.txt'

        color_img_file, train_color_label = load_visible(train_color_list)
        sketch_img_file, train_sketch_label = load_sketch(train_sketch_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)   ##原全部是144 288
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_sketch_image = []
        for i in range(len(sketch_img_file)):
            img = Image.open(data_dir+ sketch_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_sketch_image.append(pix_array)
        train_sketch_image = np.array(train_sketch_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_sketch_image = train_sketch_image
        self.train_sketch_label = train_sketch_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = sketchIndex
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_sketch = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize,
            ChannelExchange(gray=2)])

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_sketch_image[self.tIndex[index]], self.train_sketch_label[self.tIndex[index]]
        
        coin = random.randint(0, 2)
        img1 = self.transform_color(img1) if coin == 0 else self.transform_color1(img1)
        img2 = self.transform_sketch(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)   
    
def load_visible(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[-1] for s in data_file_list if s.split(' ')[-2]=='2']
        file_label = [int(s.split(' ')[1])-1 for s in data_file_list if s.split(' ')[-2]=='2']
        
    return file_image, file_label

def load_sketch(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[-1] for s in data_file_list if s.split(' ')[-2] == '1']
        file_label = [int(s.split(' ')[1]) - 1 for s in data_file_list if s.split(' ')[-2] == '1']
        
    return file_image, file_label