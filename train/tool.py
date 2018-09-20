
# coding: utf-8

import numpy as np
import torch
import os
import pickle

from PIL import Image, ImageOps, ImageEnhance
from argparse import ArgumentParser

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor, ToPILImage
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

import collections
import numbers
import random
import math

flip_index = ['16', '15', '14', '13', '12', '11', '10']

def find_bound(img):
    ''' input: numpy array
    '''
    h, w = img.shape
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sum_x = img.sum(axis=0)
    sum_y = img.sum(axis=1)
    for i in range(w):
        if sum_x[i] != 0:
            xmin = i
            break
    for i in range(w-1,-1,-1):
        if sum_x[i] != 0:
            xmax = i
            break
    for i in range(h):
        if sum_y[i] != 0:
            ymin = i
            break
    for i in range(h-1,-1,-1):
        if sum_y[i] != 0:
            ymax = i
            break
    return xmin, ymin, xmax, ymax


class Random_Rotate_Crop(object):
    def __init__(self, maxAngle = 10):
        self.maxAngle = maxAngle

    def __call__(self, img_and_label):
        img, label = img_and_label
        angle = random.uniform(-self.maxAngle, self.maxAngle)

        img = img.rotate(angle)
        label = label.rotate(angle)
        
        xmin, ymin, xmax, ymax = find_bound(np.array(label))

        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        
        xmin = max(0, xmin - 50)
        ymin = max(0, ymin - 100)
        xmax = min(1918, xmax + 50)
        ymax = min(1280, ymax + 50)
        
        if ymax - ymin <= 512:
            if ymin + 512 < 1280:
                ymax = ymin + 512
            else: ymin = ymax - 512
        if xmax - xmin <= 512:
            if xmin + 512 < 1918:
                xmax = xmin + 512
            else: xmin = xmax - 512            

        img = img.crop((xmin, ymin, xmax, ymax))
        label = label.crop((xmin, ymin, xmax, ymax))

        return img, label


class RandomColor(object):
    def __init__(self, colorRange = (0.7, 1.3), brightnessRange = (0.5, 1.5), sturaRange = (0.2, 2)):
        self.colorRange = colorRange
        self.brightnessRange = brightnessRange
        self.sturaRange = sturaRange
    
    def __call__(self, img_and_label):
        img, label = img_and_label
        l, h = self.colorRange
        r,g,b = img.split()
        ratio = np.random.uniform(l, h, 3)
        r = r.point(lambda i: i * ratio[0])
        g = g.point(lambda i: i * ratio[1])
        b = b.point(lambda i: i * ratio[2])
        rgb = [r, g, b]
        random.shuffle(rgb)
        img = Image.merge("RGB", tuple(rgb))
        brightness = ImageEnhance.Brightness(img)
        b = random.uniform(self.brightnessRange[0], self.brightnessRange[1])
        s = random.uniform(self.sturaRange[0], self.sturaRange[1])
        
        img = brightness.enhance(b)
        img = ImageEnhance.Color(img).enhance(s)

        return img, label


def min_random(x):
    n = math.ceil(x / 512.)
    if n == 1:
        xmin = random.randint(0, 20)
    elif n == 2:
        xmin_range = list(range(0, 20))
        xmin_range += list(range(x - 512 - 20, x - 512))
        xmin = random.choice(xmin_range)
    elif n == 3:
        xmin_range = list(range(0, 20))
        xmin_range += list(range(512 - 20, 512 + 20))
        xmin_range += list(range(x - 512 - 20, x - 512))
        xmin = random.choice(xmin_range)
    elif n == 4:
        xmin_range = list(range(0, 20))
        xmin_range += list(range(512 - 20, 512 + 20))
        xmin_range += list(range(1024 - 20, 1024 + 20))
        xmin_range += list(range(x - 512 - 20, x - 512))
        xmin = random.choice(xmin_range)
    if xmin + 512 > x:
        xmin = x - 512
    return xmin


class RandomCrop(object):
    def __init__(self, crop_size=512):
        self.crop_size = crop_size

    def __call__(self, img_and_label):
        img, label = img_and_label
        w, h = img.size
        
#         xmin = min_random(w)
#         ymin = min_random(h)
        xmin = random.randint(0, w - self.crop_size)
        ymin = random.randint(0, h - self.crop_size)
        
        img = img.crop((xmin, ymin, xmin+self.crop_size, ymin+self.crop_size))
        label = label.crop((xmin, ymin, xmin+self.crop_size, ymin+self.crop_size))
        
        return img, label

class RandomCrop_different_size_for_image_and_label(object):
    def __init__(self, image_size=572, label_size=388):
        self.image_size = image_size
        self.label_size = label_size
        self.bound = (self.image_size - self.label_size) // 2

    def __call__(self, img_and_label):
        img, label = img_and_label
        w, h = img.size
        
        xcenter = random.randint(self.label_size // 2, w - self.label_size // 2)
        ycenter = random.randint(self.label_size // 2, h - self.label_size // 2)
        
        img = img.crop((xcenter - self.image_size // 2, ycenter - self.image_size // 2, xcenter + self.image_size // 2, ycenter + self.image_size // 2))
        label = label.crop((xcenter - self.label_size // 2, ycenter - self.label_size // 2, xcenter + self.label_size // 2, ycenter + self.label_size // 2))
        
        return img, label
    
    
class ToTensor_Label(object):
    def __call__(self, img_and_label):
        img, label = img_and_label
        img_tensor = ToTensor()(img)
        label_tensor = torch.from_numpy(np.array(label)).long().unsqueeze(0)
        return img_tensor, label_tensor

class ImageNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, img_and_label):
        img_tensor, label_tensor = img_and_label
        for t, m, s in zip(img_tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return img_tensor, label_tensor

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class Car_dataset(Dataset):

    def __init__(self, images_root, labels_root, filenames_img, transforms=None, ifFlip=False):

        self.images_root = images_root
        self.labels_root = labels_root
        self.filenames_img = filenames_img
        self.filenames_img.sort()
        self.transforms = transforms
        self.ifFlip = ifFlip

    def __getitem__(self, index):
        filename_img = self.filenames_img[index]
        filename_mask = os.path.splitext(filename_img)[0]+'_mask.gif'
        angle_idx = os.path.splitext(filename_img)[0].split('_')[-1]

        with open(os.path.join(self.images_root, filename_img), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(os.path.join(self.labels_root, filename_mask), 'rb') as f:
            label = Image.open(f).convert('P')
        if self.ifFlip and angle_idx in flip_index:
            image = ImageOps.mirror(image)
            label = ImageOps.mirror(label)

        if self.transforms is not None:
            [image, label] = self.transforms([image, label])


        return image, label

    def __len__(self):
        return len(self.filenames_img)

