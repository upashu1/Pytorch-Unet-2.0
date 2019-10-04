# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256,bilinear)
        self.up2 = up(512, 128,bilinear)
        self.up3 = up(256, 64,bilinear)
        self.up4 = up(128, 64,bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)
    
    
class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.P4 = double_conv(512,512)
        self.P3 = double_conv(256,256)
        self.P2 = double_conv(128,128)
        self.P1 = double_conv(64,64)
        
        self.up1 = up(1024, 256,bilinear)
        self.up2 = up(512, 128,bilinear)
        self.up3 = up(256, 64,bilinear)
        self.up4 = up(128, 64,bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, self.P4(x4))
        x = self.up2(x, self.P3(x3))
        x = self.up3(x, self.P2(x2))
        x = self.up4(x, self.P1(x1))
        x = self.outc(x)
        return F.sigmoid(x)


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(UNet3, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.P4 = double_conv(512,512)
        self.P3 = double_conv(256,256)
        self.P2 = double_conv(128,128)
        self.P1 = double_conv(64,64)
        
        self.P6 = double_conv(512,512)
        self.P5 = double_conv(256,256)
        
        self.up1 = up(1024, 256,bilinear)
        self.up2 = up(512, 128,bilinear)
        self.up3 = up(256, 64,bilinear)
        self.up4 = up(128, 64,bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, self.P6(self.P4(x4)))
        x = self.up2(x, self.P5(self.P3(x3)))
        x = self.up3(x, self.P2(x2))
        x = self.up4(x, self.P1(x1))
        x = self.outc(x)
        return F.sigmoid(x)