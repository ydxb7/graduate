import torch
from torch import nn
import torch.nn.functional as F

class UNet_check(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super(UNet_check, self).__init__()

        down1 = []
        down1.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=0))
        down1.append(nn.ReLU())
        down1.append(nn.BatchNorm2d(64))
        down1.append(nn.Conv2d(64, 64, kernel_size=3, padding=0))
        down1.append(nn.ReLU())
        down1.append(nn.BatchNorm2d(64))
        self.down1 = nn.Sequential(*down1)
        
        down2 = []
        down2.append(nn.Conv2d(64, 128, kernel_size=3, padding=0))
        down2.append(nn.ReLU())
        down2.append(nn.BatchNorm2d(128))
        down2.append(nn.Conv2d(128, 128, kernel_size=3, padding=0))
        down2.append(nn.ReLU())
        down2.append(nn.BatchNorm2d(128))
        self.down2 = nn.Sequential(*down2)
        
        down3 = []
        down3.append(nn.Conv2d(128, 256, kernel_size=3, padding=0))
        down3.append(nn.ReLU())
        down3.append(nn.BatchNorm2d(256))
        down3.append(nn.Conv2d(256, 256, kernel_size=3, padding=0))
        down3.append(nn.ReLU())
        down3.append(nn.BatchNorm2d(256))
        self.down3 = nn.Sequential(*down3)
        
        down4 = []
        down4.append(nn.Conv2d(256, 512, kernel_size=3, padding=0))
        down4.append(nn.ReLU())
        down4.append(nn.BatchNorm2d(512))
        down4.append(nn.Conv2d(512, 512, kernel_size=3, padding=0))
        down4.append(nn.ReLU())
        down4.append(nn.BatchNorm2d(512))
        self.down4 = nn.Sequential(*down4)
        
        down5 = []
        down5.append(nn.Conv2d(512, 1024, kernel_size=3, padding=0))
        down5.append(nn.ReLU())
        down5.append(nn.BatchNorm2d(1024))
        down5.append(nn.Conv2d(1024, 1024, kernel_size=3, padding=0))
        down5.append(nn.ReLU())
        down5.append(nn.BatchNorm2d(1024))
        self.down5 = nn.Sequential(*down5)
        
        self.up4_x = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        up4 = []
        up4.append(nn.Conv2d(1024, 512, kernel_size=3, padding=0))
        up4.append(nn.ReLU())
        up4.append(nn.BatchNorm2d(512))
        up4.append(nn.Conv2d(512, 512, kernel_size=3, padding=0))
        up4.append(nn.ReLU())
        up4.append(nn.BatchNorm2d(512))
        self.up4 = nn.Sequential(*up4)
        
        self.up3_x = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        up3 = []
        up3.append(nn.Conv2d(512, 256, kernel_size=3, padding=0))
        up3.append(nn.ReLU())
        up3.append(nn.BatchNorm2d(256))
        up3.append(nn.Conv2d(256, 256, kernel_size=3, padding=0))
        up3.append(nn.ReLU())
        up3.append(nn.BatchNorm2d(256))
        self.up3 = nn.Sequential(*up3)
        
        self.up2_x = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        up2 = []
        up2.append(nn.Conv2d(256, 128, kernel_size=3, padding=0))
        up2.append(nn.ReLU())
        up2.append(nn.BatchNorm2d(128))
        up2.append(nn.Conv2d(128, 128, kernel_size=3, padding=0))
        up2.append(nn.ReLU())
        up2.append(nn.BatchNorm2d(128))
        self.up2 = nn.Sequential(*up2)
        
        self.up1_x = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        up1 = []
        up1.append(nn.Conv2d(128, 64, kernel_size=3, padding=0))
        up1.append(nn.ReLU())
        up1.append(nn.BatchNorm2d(64))
        up1.append(nn.Conv2d(64, 64, kernel_size=3, padding=0))
        up1.append(nn.ReLU())
        up1.append(nn.BatchNorm2d(64))
        self.up1 = nn.Sequential(*up1)
        
        self.last = nn.Conv2d(64, n_classes, kernel_size=1)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]


    def forward(self, x):
        down1 = self.down1(x)
        x = F.avg_pool2d(down1, 2)
        
        down2 = self.down2(x)
        x = F.avg_pool2d(down2, 2)
        
        down3 = self.down3(x)
        x = F.avg_pool2d(down3, 2)
        
        down4 = self.down4(x)
        x = F.avg_pool2d(down4, 2)
        
        x = self.down5(x)
        
        x = self.up4_x(x)
        crop4 = self.center_crop(down4, x.shape[2:])
        x = torch.cat([x, crop4], 1)
        x = self.up4(x)
        
        x = self.up3_x(x)
        crop3 = self.center_crop(down3, x.shape[2:])
        x = torch.cat([x, crop3], 1)
        x = self.up3(x)
        
        x = self.up2_x(x)
        crop2 = self.center_crop(down2, x.shape[2:])
        x = torch.cat([x, crop2], 1)
        x = self.up2(x)
        
        x = self.up1_x(x)
        crop1 = self.center_crop(down1, x.shape[2:])
        x = torch.cat([x, crop1], 1)
        x = self.up1(x)
        
        x = self.last(x)
    
        return x

