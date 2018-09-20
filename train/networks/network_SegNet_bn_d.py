
# coding: utf-8

# In[11]:


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


# In[12]:


class SegNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super(SegNetDec, self).__init__()

        layers = [
#             nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


# In[19]:


class SegNet_bn_enc4(nn.Module):

    def __init__(self, num_classes):
        super(SegNet_bn_enc4, self).__init__()

        # should be vgg16bn but at the moment we have no pretrained bn models
        encoders = list(models.vgg16_bn(pretrained=True).features.children())

        self.enc1 = nn.Sequential(*encoders[:7])     # 512*512*3 --> 256*256*64
        self.enc2 = nn.Sequential(*encoders[7:14])   # 256*256*64  --> 128*128*128
        self.enc3 = nn.Sequential(*encoders[14:24])  # 128*128*128  --> 64*64*256
        
        for m in self.modules():
#             if isinstance(m, nn.Conv2d):
            m.requires_grad = False
        
        self.enc4 = nn.Sequential(*encoders[24:34])  # 64*64*256 --> 32*32*512
        
        self.dec4 = SegNetDec(512, 256, 1)           # 32*32*512 --> 64*64*512
        self.dec3 = SegNetDec(512, 128, 1)          # 64*64*512 --> 128*128*128
        self.dec2 = SegNetDec(256, 64, 1)           # 128*128*256 --> 256*256*64
        self.dec1 = SegNetDec(128, 64, 1)           # 256*256*256 --> 512*512*64
        
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        enc1 = self.enc1(x)     # 512*512*3 --> 256*256*64
        enc2 = self.enc2(enc1)  # 256*256*64  --> 128*128*128
        enc3 = self.enc3(enc2)  # 128*128*128  --> 64*64*256
        enc4 = self.enc4(enc3)  # 64*64*256 --> 32*32*512
        dec4 = self.dec4(enc4)  # 32*32*512 --> 64*64*256
        dec3 = self.dec3(torch.cat([enc3, dec4], 1)) # 256+256 --> 128
        dec2 = self.dec2(torch.cat([enc2, dec3], 1)) # 128+128 --> 64
        dec1 = self.dec1(torch.cat([enc1, dec2], 1)) # 64+64 --> 64
        
#         return F.upsample_bilinear(self.final(dec1), x.size()[2:])
        return F.upsample(input = self.final(dec1), size = x.size()[2:], mode = 'bilinear', align_corners=True)

