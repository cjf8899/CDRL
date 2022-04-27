import torch.nn as nn
import torch.nn.functional as F
import torch
from cbam import *


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("BasicConv") != -1:
        torch.nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bn.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bn.bias.data, 0.0)
    elif classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#       U-NET CBAM ver
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class UNetUp_CBAM(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp_CBAM, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.ChannelGate = ChannelGate(out_size, 16, ['avg', 'max'])
        self.SpatialGate = SpatialGate()
    def forward(self, x, A, B):
        x = self.model(x)
        A = self.ChannelGate(A)
        B = self.SpatialGate(B)
        skip_input = A+B
        x = torch.cat((x, skip_input), 1)
        
        return x

class GeneratorUNet_CBAM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet_CBAM, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp_CBAM(512, 512, dropout=0.5)
        self.up2 = UNetUp_CBAM(1024, 512, dropout=0.5)
        self.up3 = UNetUp_CBAM(1024, 512, dropout=0.5)
        self.up4 = UNetUp_CBAM(1024, 512, dropout=0.5)
        self.up5 = UNetUp_CBAM(1024, 256)
        self.up6 = UNetUp_CBAM(512, 128)
        self.up7 = UNetUp_CBAM(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, xt1, xt2):
        # Images Harmonization
        
        d1_xt1 = self.down1(xt1)
        d2_xt1 = self.down2(d1_xt1)
        d3_xt1 = self.down3(d2_xt1)
        d4_xt1 = self.down4(d3_xt1)
        d5_xt1 = self.down5(d4_xt1)
        d6_xt1 = self.down6(d5_xt1)
        d7_xt1 = self.down7(d6_xt1)
        d8_xt1 = self.down8(d7_xt1)
        
        d1_xt2 = self.down1(xt2)
        d2_xt2 = self.down2(d1_xt2)
        d3_xt2 = self.down3(d2_xt2)
        d4_xt2 = self.down4(d3_xt2)
        d5_xt2 = self.down5(d4_xt2)
        d6_xt2 = self.down6(d5_xt2)
        d7_xt2 = self.down7(d6_xt2)
        d8_xt2 = self.down8(d7_xt2)
        
        d8 = d8_xt1 + d8_xt2
        
        
        u1 = self.up1(d8, d7_xt1, d7_xt2)   
        u2 = self.up2(u1, d6_xt1, d6_xt2)
        u3 = self.up3(u2, d5_xt1, d5_xt2)
        u4 = self.up4(u3, d4_xt1, d4_xt2)
        u5 = self.up5(u4, d3_xt1, d3_xt2)
        u6 = self.up6(u5, d2_xt1, d2_xt2)
        u7 = self.up7(u6, d1_xt1, d1_xt2)

        return self.final(u7)

    
    
##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    
    
    
