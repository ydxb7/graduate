import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels,  momentum=0.1),
            nn.LeakyReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels,  momentum=0.1),
            nn.LeakyReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels,  momentum=0.1),
            nn.LeakyReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)

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

class SegUNet_new_version_coord_2channel_at_middle(nn.Module):
    def __init__(self,num_classes):
        super(SegUNet_new_version_coord_2channel_at_middle, self).__init__()

        batchNorm_momentum = 0.1
        
        encoders = list(models.vgg16_bn(pretrained=True).features.children())
        
        self.enc1 = nn.Sequential(*encoders[:6])     # 512*512*3 --> 256*256*64(after maxpool)
        self.enc2 = nn.Sequential(*encoders[7:13])   # 256*256*64  --> 128*128*128(after maxpool)
        self.enc3 = nn.Sequential(*encoders[14:23])  # 128*128*128  --> 64*64*256(after maxpool)
        self.enc4 = nn.Sequential(*encoders[24:33])  # 64*64*256 --> 32*32*512(after maxpool)
        self.enc5 = nn.Sequential(*encoders[34:43])    # 32*32*512 --> 32*32*512
        
        self.dec5 = ConvBlock(512+2, 512, 1)          # 32*32*512 --> 32*32*512
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
        enc4_pool, id4 = F.max_pool2d(enc4,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        # 32*32*512 --> 16*16*512
        enc5 = self.enc5(enc4_pool) # 32*32*512
#         enc5_pool, id5 = F.max_pool2d(enc5,kernel_size=2, stride=2,return_indices=True)
        
        #-------------------------------Decoder------------------------------------------------
        # Stage 5d
        # 32*32*512 --> 32*32*512
        dec5 = self.dec5(torch.cat([enc5, coord], 1)) 
#         dec5 = self.dec5(enc5) 

        # Stage 4d
        # 32*32*512+512 --> 64*64*256
        dec5_up = F.max_unpool2d(dec5, id4, kernel_size=2, stride=2) # 32*32*512 --> 64*64*512
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
