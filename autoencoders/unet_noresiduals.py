""" Full assembly of the parts to form the complete network """

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Hidden(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.hidden_in = nn.Linear(1024*16, 500)
        self.hidden_in2 = nn.Linear(500, 100)
        self.hidden_out1 = nn.Linear(100, 500)
        self.hidden_out2 = nn.Linear(500, 1024*16)
        self.conv_transpose = nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=4)

    def forward(self, x):
        x = self.input_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.hidden_in(x)
        x = self.hidden_in2(x)
        hidden = x
        x = self.hidden_out1(x)
        x = self.hidden_out2(x)
        x = x.reshape((len(x), 1024, 4, 4))
        x = self.conv_transpose(x)
        return x, hidden

class Hidden_128(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.hidden_in = nn.Linear(1024*4, 500)
        self.hidden_in2 = nn.Linear(500, 100)
        self.hidden_out1 = nn.Linear(100, 500)
        self.hidden_out2 = nn.Linear(500, 1024*4)
        self.conv_transpose = nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=4)

    def forward(self, x):
        x = self.input_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.hidden_in(x)
        x = self.hidden_in2(x)
        hidden = x
        x = self.hidden_out1(x)
        x = self.hidden_out2(x)
        x = x.reshape((len(x), 1024, 2, 2))
        x = self.conv_transpose(x)
        return x, hidden


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x1], dim=1)
        return self.conv(x)


class NoResidualUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

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
        logits = self.outc(x)
        return logits #, x5

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet_hidden(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.hidden = Hidden_128()
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5, hidden = self.hidden(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits #, hidden

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)




class UNetWide(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 128))
        self.down1 = (Down(128, 256))
        self.down2 = (Down(256, 512))
        self.down3 = (Down(512, 1024))
        factor = 2 if bilinear else 1
        self.down4 = (Down(1024, 2048 // factor))
        self.up1 = (Up(2048, 1024 // factor, bilinear))
        self.up2 = (Up(1024, 512 // factor, bilinear))
        self.up3 = (Up(512, 256 // factor, bilinear))
        self.up4 = (Up(256, 128, bilinear))
        self.outc = (OutConv(128, n_classes))

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
        logits = self.outc(x)
        return logits #, x5

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class UNetDeep(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = False
        factor = 2

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.down5 = Down(1024, 2048)
        self.up0 =  NoResidualUp(2048, 1024)
        self.up1 = (NoResidualUp(1024, 512, bilinear))
        self.up2 = (NoResidualUp(512, 256, bilinear))
        self.up3 = (NoResidualUp(256, 128, bilinear))
        self.up4 = (NoResidualUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits 

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.up0 = torch.utils.checkpoint(self.up0)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNetDeepWide(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = False
        factor = 2

        self.inc = (DoubleConv(n_channels, 128))
        self.down1 = (Down(128, 256))
        self.down2 = (Down(256, 512))
        self.down3 = (Down(512, 1024))
        self.down4 = (Down(1024, 2048))
        self.down5 = Down(2048, 4096)
        self.up0 =  NoResidualUp(4096, 2048)
        self.up1 = (NoResidualUp(2048, 1024, bilinear))
        self.up2 = (NoResidualUp(1024, 512, bilinear))
        self.up3 = (NoResidualUp(512, 256, bilinear))
        self.up4 = (NoResidualUp(256, 128, bilinear))
        self.outc = (OutConv(128, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits 

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.up0 = torch.utils.checkpoint(self.up0)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class WideHidden(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(2048, 256, kernel_size=4, stride=2, padding=1)
        self.hidden_in = nn.Linear(4096, 10000)
        self.hidden_out = nn.Linear(10000, 4096)
        self.conv_transpose = nn.ConvTranspose2d(256, 2048, kernel_size=4, stride=4)

    def forward(self, x):
        x = self.input_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.hidden_in(x)
        hidden = x
        x = self.hidden_out(x)
        x = x.reshape((len(x), 256, 4, 4)) 
        x = self.conv_transpose(x)
        return x #, hidden

class UNetWideHidden(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 128))
        self.down1 = (Down(128, 256))
        self.down2 = (Down(256, 512))
        self.down3 = (Down(512, 1024))
        factor = 2 if bilinear else 1

        self.hidden = WideHidden()
        self.down4 = (Down(1024, 2048 // factor))
        self.up1 = (Up(2048, 1024 // factor, bilinear))
        self.up2 = (Up(1024, 512 // factor, bilinear))
        self.up3 = (Up(512, 256 // factor, bilinear))
        self.up4 = (Up(256, 128, bilinear))
        self.outc = (OutConv(128, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.hidden(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits #, x5

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)