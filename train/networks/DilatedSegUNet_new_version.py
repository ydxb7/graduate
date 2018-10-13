import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class SegUNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(SegUNetConvBlock, self).__init__()
        layers = [
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2,  momentum=0.1),
            nn.LeakyReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2,  momentum=0.1),
            nn.LeakyReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels,  momentum=0.1),
            nn.LeakyReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)

class ConvBNActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1,
                 activation=nn.ReLU(inplace=True)):
        super(ConvBNActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DilatedSegUNet_new_version(nn.Module):
    def __init__(self,num_classes=2, bottleneck_depth=6):
        super(DilatedSegUNet_new_version, self).__init__()

        batchNorm_momentum = 0.1
        
        encoders = list(models.vgg16_bn(pretrained=True).features.children())
        
        self.enc1 = nn.Sequential(*encoders[:6])     # 512*512*3 --> 256*256*64(after maxpool)
        self.enc2 = nn.Sequential(*encoders[7:13])   # 256*256*64  --> 128*128*128(after maxpool)
        self.enc3 = nn.Sequential(*encoders[14:23])  # 128*128*128  --> 64*64*256(after maxpool)
        self.enc4 = nn.Sequential(*encoders[24:33])  # 64*64*256 --> 32*32*512(after maxpool)
        
        self.bottleneck_path = nn.ModuleList()
        for i in range(bottleneck_depth):
            if i == 0:
                self.bottleneck_path.append(ConvBNActivation(512+2, 512, 3, dilation=2**i, padding=2**i))
            else:
                self.bottleneck_path.append(ConvBNActivation(512, 512, 3, dilation=2**i, padding=2**i))
           
        self.dec4 = SegUNetConvBlock(1024, 256, 1)          # 32*32*512+512(after up) --> 64*64*256
        self.dec3 = SegUNetConvBlock(512, 128, 1)           # 64*64*256+256(after up) --> 128*128*128
        self.dec2 = SegUNetConvBlock(256, 64, 1)            # 128*128*128+128(after up) --> 256*256*64
        self.dec1 = SegUNetConvBlock(128, 64, 1)            # 256*256*64+64(after up) --> 512*512*64
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)


    def forward(self, x, coord):
        #--------------------------------Encoder-------------------------------------------
        # Stage 1
        # 512*512*3 --> 256*256*64
        enc1 = self.enc1(x) # 512*512*64
        enc1_pool, id1 = F.max_pool2d(enc1, kernel_size=2, stride=2,return_indices=True)     

        # Stage 2
        # 256*256*64  --> 128*128*128
        enc2 = self.enc2(enc1_pool) # 256*256*128
        enc2_pool, id2 = F.max_pool2d(enc2, kernel_size=2, stride=2,return_indices=True) 
        
        # Stage 3
        # 128*128*128  --> 64*64*256
        enc3 = self.enc3(enc2_pool) # 128*128*256
        enc3_pool, id3 = F.max_pool2d(enc3,kernel_size=2, stride=2,return_indices=True)
        
        # Stage 4
        # 64*64*256 --> 32*32*512
        enc4 = self.enc4(enc3_pool) # 64*64*512
        x, id4 = F.max_pool2d(enc4,kernel_size=2, stride=2,return_indices=True)

        dilated_layers = []
        for i, bneck in enumerate(self.bottleneck_path):
            if i == 0:
                x = bneck(torch.cat([x, coord], 1))
            else:
                x = bneck(x)
            dilated_layers.append(x.unsqueeze(-1))
        x = torch.cat(dilated_layers, dim=-1)
        x = torch.sum(x, dim=-1)
        

        # Stage 4d
        # 32*32*512+512 --> 64*64*256
        dec5_up = F.max_unpool2d(x, id4, kernel_size=2, stride=2) # 32*32*512 --> 64*64*512
        dec4 = self.dec4(torch.cat([enc4, dec5_up], 1)) # 64*64*(512+512) --> 64*64*256

        # Stage 3d
        dec4_up = F.max_unpool2d(dec4, id3, kernel_size=2, stride=2) # 64*64*256 --> 128*128*256
        dec3 = self.dec3(torch.cat([enc3, dec4_up], 1)) # 128*128*(256+256) --> 128*128*128

        # Stage 2d
        dec3_up = F.max_unpool2d(dec3, id2, kernel_size=2, stride=2) # 128*128*128 --> 256*256*128
        dec2 = self.dec2(torch.cat([enc2, dec3_up], 1)) # 256*256*(128+128) --> 256*256*64

        # Stage 1d
        dec2_up = F.max_unpool2d(dec2, id1, kernel_size=2, stride=2) # 256*256*64 --> 512*512*64
        dec1 = self.dec1(torch.cat([enc1, dec2_up], 1)) # 512*512*(64+64) --> 512*512*64

        final = self.final(dec1)
        
        return final
