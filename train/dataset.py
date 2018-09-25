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
from scipy.ndimage.interpolation import rotate

flip_index = ['16', '15', '14', '13', '12', '11', '10']

class Car_dataset(Dataset):

    def __init__(self, images_root, labels_root, filenames_img, Flip_to_same_side=False, ifcoord=False,
                 randomColor=False, randomCrop=False, crop_size=512, ifrotate=False, maxangle=5, stride=32):

        self.images_root = images_root
        self.labels_root = labels_root
        self.filenames_img = filenames_img
        self.filenames_img.sort()
        self.Flip_to_same_side = Flip_to_same_side
        self.ifcoord = ifcoord
        self.randomColor = randomColor
        self.randomCrop = randomCrop
        self.crop_size = crop_size
        self.ifrotate = ifrotate
        self.maxangle = maxangle
        self.stride = stride

    def __getitem__(self, index):
        filename_img = self.filenames_img[index]
        filename_mask = os.path.splitext(filename_img)[0]+'_mask.gif'
        angle_idx = os.path.splitext(filename_img)[0].split('_')[-1]

        with open(os.path.join(self.images_root, filename_img), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(os.path.join(self.labels_root, filename_mask), 'rb') as f:
            label = Image.open(f).convert('P')
        if self.Flip_to_same_side and angle_idx in flip_index:
            image = ImageOps.mirror(image)
            label = ImageOps.mirror(label)
            
        if self.randomColor:
            image = self.RandomColor(image)
            
        if not hasattr(self, 'w'):
            self.w, self.h = image.size
            
        images = [np.array(image), np.array(label)]
        
        if self.ifcoord:
            xx,yy = np.meshgrid(np.linspace(-0.5,0.5,w), np.linspace(-0.5,0.5,h))
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...]],0).astype('float32')
            images.append(coord)

        images[0] = images[0].transpose(2, 0, 1)
        
        if self.ifrotate:
            images = self.Rotate(images)

        if self.randomCrop:
            images = self.RandomCrop(images)

        if self.ifcoord:
            images[2] = images[2][:, ::self.stride, ::self.stride]
            
        images[0] = self.ImageNormalize(images[0])

        return images[0], images[1], images[2]
        

    def __len__(self):
        return len(self.filenames_img)


    def RandomColor(self, img, colorRange = (0.7, 1.3), brightnessRange = (0.5, 1.5), sturaRange = (0.2, 2)):
        l, h = colorRange
        r,g,b = img.split()
        ratio = np.random.uniform(l, h, 3)
        r = r.point(lambda i: i * ratio[0])
        g = g.point(lambda i: i * ratio[1])
        b = b.point(lambda i: i * ratio[2])
        rgb = [r, g, b]
        random.shuffle(rgb)
        img = Image.merge("RGB", tuple(rgb))
        brightness = ImageEnhance.Brightness(img)
        b = random.uniform(brightnessRange[0], brightnessRange[1])
        s = random.uniform(sturaRange[0], sturaRange[1])
        
        img = brightness.enhance(b)
        img = ImageEnhance.Color(img).enhance(s)
        return img

    def RandomCrop(self, images):
        xmin = random.randint(0, self.w - self.crop_size)
        ymin = random.randint(0, self.h - self.crop_size)
        if self.ifcoord:
            [image, label, coord] = images
            image = image[:, ymin:ymin+self.crop_size, xmin:xmin+self.crop_size]
            coord = coord[:, ymin:ymin+self.crop_size, xmin:xmin+self.crop_size]
            label = label[ymin:ymin+self.crop_size, xmin:xmin+self.crop_size]
            return [image, label, coord]
        else:
            [image, label] = images
            images = images[:, ymin:ymin+self.crop_size, xmin:xmin+self.crop_size]
            label = label[ymin:ymin+self.crop_size, xmin:xmin+self.crop_size]
            return [image, label]

    def Rotate(self, images):
        angle1 = np.random.rand() * self.maxangle
        if self.ifcoord:
            coord = rotate(images[2],angle1,axes=(1, 2),reshape=False)
        image = rotate(images[0],angle1,axes=(1, 2),reshape=False)
        label = rotate(images[1],angle1,reshape=False)
        return [image, label, coord]
    
    def ImageNormalize(self, image):
        image = image.astype('float32') / 255
        mean = [.485, .456, .406]
        std = [.229, .224, .225]
        for i in range(3):
            image[i] = (image[i] - mean[i]) / std[i]
        return image
