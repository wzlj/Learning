import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from modelbackbone import build_backbone

import resnet

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_1x1 = nn.Conv2d(in_ch, in_ch//2, 1, padding=1)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        if in_ch != out_ch:
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.conv = double_conv(in_ch*2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.size()[1] != x2.size()[1]:
            x1 = self.conv_1x1(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)#输出通道
        return x


# sp_up(256, 64)
class sp_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(sp_up, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, 1, padding=1)
        self.conv = double_conv(out_ch*2, out_ch)

    def forward(self, x1, x2):

        x1 = self.conv_1x1(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)  # 输出通道
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, backbone='resnet50'):
        super(UNet, self).__init__()
        # self.backbone = build_backbone(backbone, output_stride=16, BatchNorm=nn.BatchNorm2d)#不使用sync_bn
        self.backbone = resnet.ResNet101(output_stride=16, BatchNorm=nn.BatchNorm2d)
        # [, 64, 128, 128]
        # self.inc = inconv(n_channels, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = sp_up(256, 64)
        self.up4 = up(64, 64)
        self.up5 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.backbone.conv1(x)  # [, 3, 256, 256]
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)  # [, 64, 128, 128]
        x2 = self.backbone.maxpool(x1)  # [, 64, 64, 64]

        x3 = self.backbone.layer1(x2)  # [, 256, 64, 64]
        x4 = self.backbone.layer2(x3)  # [, 512, 32, 32]
        x5 = self.backbone.layer3(x4)  # [, 1024, 16, 16]
        # x5 = self.backbone.layer4(x4)

        _x = self.up1(x5, x4)  # 16 --> 32
        _x = self.up2(_x, x3)  # 32 --> 64
        _x = self.up3(_x, x2)  # 64 --> 64
        _x = self.up4(_x, x1)  # 64 --> 128
        _x = self.up5(_x)  # 128 --> 256
        x = self.outc(_x)
        return F.sigmoid(x)  # 结尾处没有使用sigmoid

    # def get_1x_lr_params(self):
    #     modules = [self.backbone]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
    #                     or isinstance(m[1], nn.BatchNorm2d):
    #                 for p in m[1].parameters():
    #                     if p.requires_grad:
    #                         yield p
    #
    # def get_10x_lr_params(self):
    #     modules = [self.up1, self.up2, self.up3, self.up3, self.up4, self.outc]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
    #                     or isinstance(m[1], nn.BatchNorm2d):
    #                 for p in m[1].parameters():
    #                     if p.requires_grad:
    #                         yield p

if __name__ == '__main__':
    model = UNet(3, 4).cuda()
    input_arr = torch.randn(4, 3, 256, 256).cuda()
    out = model(input_arr)
    print(out.shape)
