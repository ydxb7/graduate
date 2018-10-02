import torch
from torch import nn
import torch.nn.functional as F

class UNet_check(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super(UNet_check, self).__init__()

        # 572->568
        down1 = []
        down1.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=0))
        down1.append(nn.ReLU())
        down1.append(nn.BatchNorm2d(64))
        down1.append(nn.Conv2d(64, 64, kernel_size=3, padding=0))
        down1.append(nn.ReLU())
        down1.append(nn.BatchNorm2d(64))
        self.down1 = nn.Sequential(*down1)
        
        # 284->280
        down2 = []
        down2.append(nn.Conv2d(64, 128, kernel_size=3, padding=0))
        down2.append(nn.ReLU())
        down2.append(nn.BatchNorm2d(128))
        down2.append(nn.Conv2d(128, 128, kernel_size=3, padding=0))
        down2.append(nn.ReLU())
        down2.append(nn.BatchNorm2d(128))
        self.down2 = nn.Sequential(*down2)
        
        # 140->136
        down3 = []
        down3.append(nn.Conv2d(128, 256, kernel_size=3, padding=0))
        down3.append(nn.ReLU())
        down3.append(nn.BatchNorm2d(256))
        down3.append(nn.Conv2d(256, 256, kernel_size=3, padding=0))
        down3.append(nn.ReLU())
        down3.append(nn.BatchNorm2d(256))
        self.down3 = nn.Sequential(*down3)
        
        # 68->64
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

    def forward(self, x):
        down1 = self.down1(x) # 568
        x = F.max_pool2d(down1, kernel_size=2, stride=2)
        
        down2 = self.down2(x) # 280
        x = F.max_pool2d(down2, kernel_size=2, stride=2)
        
        down3 = self.down3(x) # 136
        x = F.max_pool2d(down3, kernel_size=2, stride=2)
        
        down4 = self.down4(x) # 64
        x = F.max_pool2d(down4, kernel_size=2, stride=2)
        
        x = self.down5(x) # 28
        
        x = self.up4_x(x) # 56
        down4_crop = down4[:, :, 4:-4, 4:-4] # 64->56
        x = self.up4(torch.cat([down4_crop, x], 1)) # 52
        
        x = self.up3_x(x) # 104
        down3_crop = down3[:, :, 16:-16, 16:-16] # 136->104
        x = self.up3(torch.cat([down3_crop, x], 1)) # 100
        
        x = self.up2_x(x) # 200
        down2_crop = down2[:, :, 40:-40, 40:-40] # 280->200
        x = self.up2(torch.cat([down2_crop, x], 1)) # 196
        
        x = self.up1_x(x) # 392
        down1_crop = down1[:, :, 88:-88, 88:-88] # 568->392
        x = self.up1(torch.cat([down1_crop, x], 1)) # 388
        
        x = self.last(x)
    
        return x
