import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.pvtv2 import pvt_v2_b2


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x





class RFEM(nn.Module):
    def __init__(self,
                 channels_1,
                 channels_2,
                 channels
                 ):
        super(RFEM, self).__init__()
        self.conv1_1 = Conv1x1(channels_1 + channels_2, channels)
        self.conv1_1x1 = nn.Conv2d(3 * channels // 2, channels // 4, 1)
        self.conv2_1x1 = nn.Conv2d(7 * channels // 4, channels // 4, 1)
        self.conv3_1x1 = nn.Conv2d(2 * channels, channels, 1)
        self.dconv_r2 = ConvBNReLU(in_channels=channels, out_channels=channels // 2, kernel_size=3, dilation=2)
        self.dconv_r4 = ConvBNReLU(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, dilation=4)
        self.dconv_r8 = ConvBNReLU(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, dilation=8)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x=self.conv1_1(x)
        r1 = self.dconv_r2(x)
        r1 = torch.cat([r1, x], dim=1)

        r2 = self.dconv_r4(self.conv1_1x1(r1))
        r2 = torch.cat([r2, r1], dim=1)

        r3 = self.dconv_r8(self.conv2_1x1(r2))
        r3 = torch.cat([r3, r2], dim=1)

        out = self.conv3_1x1(r3)

        return out


    # def forward(self, input):
    #     R1 = self.dconv_r2(input)
    #     R1 = torch.cat([R1, input], dim=1)
    #
    #     R2 = self.dconv_r4(self.conv1_1x1(R1))
    #     R2 = torch.cat([R2, R1], dim=1)
    #
    #     R3 = self.dconv_r8(self.conv2_1x1(R2))
    #     R3 = torch.cat([R3, R2], dim=1)
    #
    #     out = self.conv3_1x1(R3)
    #
    #     return out

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义主干
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/home/hy/project/xyx/MyNet_BGNet/models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # self.reduce1 = Conv1x1(64, 1)
        # self.reduce2 = Conv1x1(128, 128)
        # self.reduce3 = Conv1x1(320, 256)
        # self.reduce4 = Conv1x1(512, 256)

        self.refm1 = RFEM(128, 64,64)
        self.refm2 = RFEM(320, 128,128)
        self.refm3 = RFEM(512, 320,320)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        # self.predictor2 = nn.Conv2d(128, 1, 1)
        # self.predictor3 = nn.Conv2d(320, 1, 1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)  # [64, 128, 320, 512]

        x34 = self.refm3(x3, x4)
        x234 = self.refm2(x2, x34)
        x1234 = self.refm1(x1, x234)
        #
        # o3 = self.predictor3(x34)
        # o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        # o2 = self.predictor2(x234)
        # o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        # o1 = self.predictor1(x1234)
        # o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        # oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        out = self.predictor1(x1234)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out, self.sigmoid(out)


if __name__ == '__main__':
    input = torch.rand(8, 3, 224, 224)

    model = Net()
    out1,out2= model(input)

    print(out1.shape)
    print(out2.shape)
    # print(out3.shape)
    # print(out4.shape)