
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


# In[2]:


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


# In[5]:


class SegNet(nn.Module):

    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        # should be vgg16bn but at the moment we have no pretrained bn models
        encoders = list(models.vgg16(pretrained=True).features.children())

        self.enc1 = nn.Sequential(*encoders[:5])
        self.enc2 = nn.Sequential(*encoders[5:10])
        self.enc3 = nn.Sequential(*encoders[10:17])
        self.enc4 = nn.Sequential(*encoders[17:24])
        self.enc5 = nn.Sequential(*encoders[24:])

        # gives better results
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        self.dec5 = SegNetDec(512, 512, 1)
        self.dec4 = SegNetDec(1024, 256, 1)
        self.dec3 = SegNetDec(512, 128, 1)
        self.dec2 = SegNetDec(256, 64, 0)
        self.dec1 = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        enc1 = self.enc1(x)     # 256*256*3 --> 128*128*64
        enc2 = self.enc2(enc1)  # 128*128*64 --> 64*64*128
        enc3 = self.enc3(enc2)  # 64*64*128 --> 32*32*256
        enc4 = self.enc4(enc3)  # 32*32*256 --> 16*16*512
        enc5 = self.enc5(enc4)  # 16*16*512 --> 8*8*512
        dec5 = self.dec5(enc5)                         # 8*8*512 --> 16*16*256
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))   # 16*16*1024 --> 32*32*256
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))   # 32*32*512 --> 64*64*128
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))   # 64*64*256 --> 128*128*64
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))   # 128*128*128 --> 256*256*num_classes

#         return F.upsample_bilinear(self.final(dec1), x.size()[2:])
        return F.upsample(input = self.final(dec1), size = x.size()[2:], mode = 'bilinear', align_corners=True)

