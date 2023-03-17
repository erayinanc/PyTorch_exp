"""
U-net:
https://github.com/milesial/Pytorch-UNet/blob/master/unet/
R2 Att U-net:
https://github.com/LeeJunHyun/Image_Segmentation

included models:
    - mini U-Net
    - U-Net
    - recurrent residual U-Net
    - attention U-Net
    - recurrent residual attention U-Net
    - 3D U-Net
    - recurrent residual attention 3D U-Net
"""
import time, numpy as np, scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from perlin_noise import PerlinNoise

# mini U-NET with only single bottom layer (for dev)
class mini_U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,bilinear=True):
        super().__init__()
        fac = 2 if bilinear else 1

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128 // fac)

        self.Up2 = up_conv(ch_in=128, bilinear=bilinear)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, bilinear=bilinear)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        # decoding + concat path
        d2 = self.Up2(x2)
        d2 = up_cat(x1,d2)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)

# U-NET
class U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,bilinear=True):
        super().__init__()
        fac = 2 if bilinear else 1

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024 // fac)

        self.Up5 = up_conv(ch_in=1024, bilinear=bilinear)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512 // fac, bilinear=bilinear)
        self.Up4 = up_conv(ch_in=512, bilinear=bilinear)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256 // fac, bilinear=bilinear)
        self.Up3 = up_conv(ch_in=256, bilinear=bilinear)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128 // fac, bilinear=bilinear)
        self.Up2 = up_conv(ch_in=128, bilinear=bilinear)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, bilinear=bilinear)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = up_cat(x4,d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = up_cat(x3,d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = up_cat(x2,d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = up_cat(x1,d2)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)

class conv_block(nn.Module):
    """ added bilinear threatment """
    def __init__(self,ch_in,ch_out,bilinear=False):
        super().__init__()
        ch_mid = ch_out if not bilinear else ch_in // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )

    def forward(self,x):
        return self.double_conv(x)

class up_conv(nn.Module):
    """ added bilinear threatment """
    def __init__(self,ch_in,bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_in // 2, kernel_size=2, stride=2)

    def forward(self,x):
        return self.up(x)

def up_cat(x1,x2):
    """
    fixes dimension problems with concatanate
    if you have padding issues, see
    https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    """
    return torch.cat((x1,up_pad(x1,x2)),dim=1)

def up_pad(x1,x2):
    """ fixes dimension problems with padding
    pads smaller x2 to larger x1 so x1 and x2 have the same dimensions
    """
    diffY = x1.size()[2] - x2.size()[2]
    diffX = x1.size()[3] - x2.size()[3]
    return F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

# R2 U-Net
class R2U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2,bilinear=True):
        super().__init__()
        fac = 2 if bilinear else 1

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024 // fac, t=t)

        self.Up5 = up_conv(ch_in=1024, bilinear=bilinear)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512 // fac, t=t, bilinear=bilinear)
        self.Up4 = up_conv(ch_in=512, bilinear=bilinear)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256 // fac, t=t, bilinear=bilinear)
        self.Up3 = up_conv(ch_in=256, bilinear=bilinear)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128 // fac, t=t, bilinear=bilinear)
        self.Up2 = up_conv(ch_in=128, bilinear=bilinear)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, bilinear=bilinear)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = up_cat(x4,d5)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = up_cat(x3,d4)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = up_cat(x2,d3)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = up_cat(x1,d2)
        d2 = self.Up_RRCNN2(d2)

        return self.Conv_1x1(d2)

class RRCNN_block(nn.Module):
    """ added bilinear threatment """
    def __init__(self,ch_in,ch_out,t=2,bilinear=False):
        super().__init__()
        #ch_mid = ch_out if not bilinear else ch_in // 2
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
            )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        return x+self.RCNN(x)

class Recurrent_block(nn.Module):
    """ added bilinear threatment """
    def __init__(self,ch_out,t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
	    nn.BatchNorm2d(ch_out),
	    nn.ReLU(inplace=True)
            )

    def forward(self,x):
        for i in range(self.t):
            x1 = self.conv(x) if i==0 else self.conv(x+x1)
        return x1

# Att U-Net
class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,bilinear=True):
        super().__init__()
        fac = 2 if bilinear else 1

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024 // fac)

        self.Up5 = up_conv(ch_in=1024, bilinear=bilinear)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512 // fac, bilinear=bilinear)

        self.Up4 = up_conv(ch_in=512, bilinear=bilinear)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256 // fac, bilinear=bilinear)

        self.Up3 = up_conv(ch_in=256, bilinear=bilinear)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128 // fac, bilinear=bilinear)

        self.Up2 = up_conv(ch_in=128, bilinear=bilinear)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, bilinear=bilinear)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = up_cat(x4,d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = up_cat(x3,d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = up_cat(x2,d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = up_cat(x1,d2)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        # fixes padding issues
        g = up_pad(x,g)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        return x*self.psi(psi)

# R2 Att U-Net (combine all)
class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2,bilinear=True):
        super().__init__()
        fac = 2 if bilinear else 1

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024 // fac,t=t)

        self.Up5 = up_conv(ch_in=1024, bilinear=bilinear)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512 // fac, t=t, bilinear=bilinear)
        self.Up4 = up_conv(ch_in=512, bilinear=bilinear)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256 // fac, t=t, bilinear=bilinear)
        self.Up3 = up_conv(ch_in=256, bilinear=bilinear)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128 // fac, t=t, bilinear=bilinear)
        self.Up2 = up_conv(ch_in=128, bilinear=bilinear)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, bilinear=bilinear)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = up_cat(x4,d5)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = up_cat(x3,d4)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = up_cat(x2,d3)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = up_cat(x1,d2)
        d2 = self.Up_RRCNN2(d2)

        return self.Conv_1x1(d2)

# 3D U-NET
class TDU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,trilinear=True):
        super().__init__()
        fac = 2 if trilinear else 1

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_3d(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block_3d(ch_in=64, ch_out=128)
        self.Conv3 = conv_block_3d(ch_in=128, ch_out=256)
        self.Conv4 = conv_block_3d(ch_in=256, ch_out=512)
        self.Conv5 = conv_block_3d(ch_in=512, ch_out=1024 // fac)

        self.Up5 = up_conv_3d(ch_in=1024, trilinear=trilinear)
        self.Up_conv5 = conv_block_3d(ch_in=1024, ch_out=512 // fac, trilinear=trilinear)
        self.Up4 = up_conv_3d(ch_in=512, trilinear=trilinear)
        self.Up_conv4 = conv_block_3d(ch_in=512, ch_out=256 // fac, trilinear=trilinear)
        self.Up3 = up_conv_3d(ch_in=256, trilinear=trilinear)
        self.Up_conv3 = conv_block_3d(ch_in=256, ch_out=128 // fac, trilinear=trilinear)
        self.Up2 = up_conv_3d(ch_in=128, trilinear=trilinear)
        self.Up_conv2 = conv_block_3d(ch_in=128, ch_out=64, trilinear=trilinear)

        self.Conv_1x1 = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = up_cat_3d(x4,d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = up_cat_3d(x3,d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = up_cat_3d(x2,d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = up_cat_3d(x1,d2)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)

class conv_block_3d(nn.Module):
    """ conv_block with 3D fields """
    def __init__(self,ch_in,ch_out,trilinear=False):
        super().__init__()
        ch_mid = ch_out if not trilinear else ch_in // 2
        self.double_conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_mid, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_mid, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
            )

    def forward(self,x):
        return self.double_conv(x)

class up_conv_3d(nn.Module):
    """ conv_block with 3D fields """
    def __init__(self,ch_in,trilinear=False):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(ch_in, ch_in // 2, kernel_size=2, stride=2)

    def forward(self,x):
        return self.up(x)

def up_cat_3d(x1,x2):
    """ conv_block with 3D fields """
    return torch.cat((x1,up_pad_3d(x1,x2)),dim=1)

def up_pad_3d(x1,x2):
    """ conv_block with 3D fields """
    diffZ = x1.size()[2] - x2.size()[2]
    diffY = x1.size()[3] - x2.size()[3]
    diffX = x1.size()[4] - x2.size()[4]

    return F.pad(x2, [diffX // 2, diffX - diffX // 2, \
                      diffY // 2, diffY - diffY // 2, \
                      diffZ // 2, diffZ - diffZ // 2])

# R2 Att 3D U-Net (combine all w/ 3d conv)
class R2Att3DU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2,trilinear=True):
        super().__init__()
        fac = 2 if trilinear else 1

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.RRCNN1 = RRCNN_block_3d(ch_in=img_ch,ch_out=64,t=t)
        self.RRCNN2 = RRCNN_block_3d(ch_in=64,ch_out=128,t=t)
        self.RRCNN3 = RRCNN_block_3d(ch_in=128,ch_out=256,t=t)
        self.RRCNN4 = RRCNN_block_3d(ch_in=256,ch_out=512,t=t)
        self.RRCNN5 = RRCNN_block_3d(ch_in=512,ch_out=1024 // fac,t=t)

        self.Up5 = up_conv_3d(ch_in=1024, trilinear=trilinear)
        self.Att5 = Attention_block_3d(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block_3d(ch_in=1024, ch_out=512 // fac, t=t, trilinear=trilinear)
        self.Up4 = up_conv_3d(ch_in=512, trilinear=trilinear)
        self.Att4 = Attention_block_3d(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block_3d(ch_in=512, ch_out=256 // fac, t=t, trilinear=trilinear)
        self.Up3 = up_conv_3d(ch_in=256, trilinear=trilinear)
        self.Att3 = Attention_block_3d(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block_3d(ch_in=256, ch_out=128 // fac, t=t, trilinear=trilinear)
        self.Up2 = up_conv_3d(ch_in=128, trilinear=trilinear)
        self.Att2 = Attention_block_3d(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block_3d(ch_in=128, ch_out=64, t=t, trilinear=trilinear)

        self.Conv_1x1 = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = up_cat_3d(x4,d5)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = up_cat_3d(x3,d4)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = up_cat_3d(x2,d3)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = up_cat_3d(x1,d2)
        d2 = self.Up_RRCNN2(d2)

        return self.Conv_1x1(d2)

class RRCNN_block_3d(nn.Module):
    """ modified for conv_block with 3D fields """
    def __init__(self,ch_in,ch_out,t=2,trilinear=False):
        super().__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block_3d(ch_out,t=t),
            Recurrent_block_3d(ch_out,t=t)
            )
        self.Conv_1x1 = nn.Conv3d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        return x+self.RCNN(x)

class Recurrent_block_3d(nn.Module):
    """ modified for conv_block with 3D fields """
    def __init__(self,ch_out,t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
	    nn.BatchNorm3d(ch_out),
	    nn.ReLU(inplace=True)
            )

    def forward(self,x):
        for i in range(self.t):
            x1 = self.conv(x) if i==0 else self.conv(x+x1)
        return x1

class Attention_block_3d(nn.Module):
    """ modified for conv_block with 3D fields """
    def __init__(self,F_g,F_l,F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        # fixes padding issues
        g = up_pad_3d(x,g)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        return x*self.psi(psi)

# diffusion model
class diff_model_2d:
    def __init__(self,inputs,noise_map,epoch=1,sigma=None,eps=1e-2):
        # start a timer
        lt = time.perf_counter()

        # b,s,n_x,n_y
        m,n,a,b = inputs.shape[:]

        # level of diffusion over epochs (modify if sigma is not given in arg)
        if sigma is None:
            sigma = 1.0 if epoch >  0 else sigma
            sigma = 2.0 if epoch >  5 else sigma
            sigma = 3.0 if epoch > 10 else sigma
            sigma = 4.0 if epoch > 15 else sigma
            sigma = 5.0 if epoch > 20 else sigma
        self.sigma = sigma

        # generate Perlin noise only if dimensions of input changed (expensive)
        try:
            try_e = inputs[0,0,:,:] + noise_map
        except ValueError:
            noise_map = noise_gen_2d(a,b)
            if grank==0:
                logging.info('noise regenerated!')

        # apply filter with added noise for diffusion
        self.datas = torch.clone(inputs)
        if sigma>1.01:
            # 2D-Gaussian filter with sigma (size=2*r+1, w/ r=round(sigma,truncate), truncate=1 to get desired r)
            for i in range(m):
                for j in range(n):
                    res = sp.ndimage.gaussian_filter(inputs[i,j,:,:], self.sigma, truncate=1.0)
                    self.datas[i,j,:,:] = torch.from_numpy(res + noise_map*eps)

        self.et = time.perf_counter()-lt

# diffusion model with 3D convolutions
class diff_model_3d:
    """ modified for conv_block with 3D fields """
    def __init__(self,inputs,noise_map,epoch=1,sigma=None,eps=1e-2):
        # start a timer
        lt = time.perf_counter()

        # b,s,n_x,n_y,n_z
        m,n,a,b,c = inputs.shape[:]

        # level of diffusion over epochs (modify if sigma is not given in arg)
        if sigma is None:
            sigma = 1.0 if epoch >  0 else sigma
            sigma = 2.0 if epoch >  5 else sigma
            sigma = 3.0 if epoch > 10 else sigma
            sigma = 4.0 if epoch > 15 else sigma
            sigma = 5.0 if epoch > 20 else sigma
        self.sigma = sigma

        # generate Perlin noise only if dimensions of input changed (expensive)
        try:
            try_e = inputs[0,0,:,:,:] + noise_map
        except ValueError:
            noise_map = noise_gen_3d(a,b,c)
            if grank==0:
                logging.info('noise regenerated!')

        # apply filter with added noise for diffusion
        self.datas = torch.clone(inputs)
        if sigma>1.01:
            # 2D-Gaussian filter with sigma (size=2*r+1, w/ r=round(sigma,truncate), truncate=1 to get desired r)
            for i in range(m):
                for j in range(n):
                    res = sp.ndimage.gaussian_filter(inputs[i,j,:,:,:], self.sigma, truncate=1.0)
                    self.datas[i,j,:,:,:] = torch.from_numpy(res + noise_map*eps)

        self.et = time.perf_counter()-lt

# noise generator 2D
def noise_gen_2d(xpix,ypix):
    # Perlin noise
    noise = PerlinNoise(octaves=10)
    noise_map = [[noise([i/xpix, j/ypix]) for i in range(ypix)] for j in range(xpix)]
    return np.array(noise_map)

# noise generator 3D
def noise_gen_3d(xpix,ypix,zpix):
    """ Perlin noise, modified for conv_block with 3D fields """
    noise = PerlinNoise(octaves=10, seed=1)
    noise_map = [[[noise([i/xpix, j/ypix, k/zpix]) for i in range(zpix)] for j in range(ypix)] for k in range(xpix)]
    return np.array(noise_map)

# EOF