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


class CAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, high_feat, low_feat):
        x = self.up(high_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义主干
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r'models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # self.reduce1 = Conv1x1(64, 1)
        # self.reduce2 = Conv1x1(128, 128)
        # self.reduce3 = Conv1x1(320, 256)
        # self.reduce4 = Conv1x1(512, 256)

        self.cam1 = CAM(128, 64)
        self.cam2 = CAM(320, 128)
        self.cam3 = CAM(512, 320)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        # self.predictor2 = nn.Conv2d(128, 1, 1)
        # self.predictor3 = nn.Conv2d(320, 1, 1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)  # [64, 128, 320, 512]

        x34 = self.cam3(x3, x4)
        x234 = self.cam2(x2, x34)
        x1234 = self.cam1(x1, x234)
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
    input = torch.rand(8, 3, 32, 32)

    model = Net()
    out1, out2, out3, out4 = model(input)

    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)

