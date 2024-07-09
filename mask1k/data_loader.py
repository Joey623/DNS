import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math
import cv2


class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            if len(pix_array.shape) == 2:
                pix_array = cv2.cvtColor(pix_array, cv2.COLOR_GRAY2RGB)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


# I only train the single-shot. If you want to test multi-shot, please refer to https://github.com/Lin-Kayla/subjectivity-sketch-reid
class Mask1kData_single(data.Dataset):
    def __init__(self, data_dir, train_style, args, transform=None, colorIndex=None, thermalIndex=None):

        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'feature/train_rgb_img.npy')
        self.train_color_label = np.load(data_dir + 'feature/train_rgb_label.npy')
        train_sketch_image = np.load(data_dir + 'feature/train_sk_img_{}.npy'.format(train_style))
        self.train_sketch_label = np.load(data_dir + 'feature/train_sk_label_{}.npy'.format(train_style))
        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_sketch_image = train_sketch_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_sketch = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize,
            ChannelExchange(gray=2)])

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_sketch_image[self.tIndex[index]], self.train_sketch_label[self.tIndex[index]]

        coin = random.randint(0, 2)
        img1 = self.transform_color(img1) if coin == 0 else self.transform_color1(img1)
        img2 = self.transform_sketch(img2)


        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size=(144, 288)):
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
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


class TestData_ensemble(data.Dataset):
    def __init__(self, test_img_file, test_label, test_style, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            if len(pix_array.shape) == 2:
                pix_array = cv2.cvtColor(pix_array, cv2.COLOR_GRAY2RGB)
            test_image.append(pix_array)
        test_image = np.array(test_image)

        print('gall size', test_image.shape)
        print('gall label size', len(test_label))

        self.test_image = test_image
        self.test_label = test_label
        self.test_style = test_style
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        style = self.test_style[index]
        return img1, target1, style

    def __len__(self):
        return len(self.test_image)