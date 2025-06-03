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



class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        #self.reduce1 = Conv1x1(256, 64)
        self.reduce1 = Conv1x1(64, 64)
        #self.reduce4 = Conv1x1(2048, 256)
        self.reduce4 = Conv1x1(512, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out


class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

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
        path = r'models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # if self.training:
        #     self.initialize_weights()

        self.eam = EAM()

        self.efm1 = EFM(64)
        self.efm2 = EFM(128)
        self.efm3 = EFM(320)
        self.efm4 = EFM(512)

        # self.reduce1 = Conv1x1(256, 64)
        # self.reduce2 = Conv1x1(512, 128)
        # self.reduce3 = Conv1x1(1024, 256)
        # self.reduce4 = Conv1x1(2048, 256)

        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 256)
        self.reduce4 = Conv1x1(512, 256)

        self.cam1 = CAM(128, 64)
        self.cam2 = CAM(256, 128)
        self.cam3 = CAM(256, 256)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(128, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)


    # def initialize_weights(self):
    #     model_state = torch.load('models/res2net50_v1b_26w_4s-3cf99910.pth')
    #     self.resnet.load_state_dict(model_state, strict=False)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)  # [64, 128, 320, 512]



        edge = self.eam(x4, x1)
        edge_att = torch.sigmoid(edge)


        x1a = self.efm1(x1, edge_att)
        x2a = self.efm2(x2, edge_att)
        x3a = self.efm3(x3, edge_att)
        x4a = self.efm4(x4, edge_att)

        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)

        x34 = self.cam3(x3r, x4r)
        x234 = self.cam2(x2r, x34)
        x1234 = self.cam1(x1r, x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe


if __name__ == '__main__':
    input = torch.rand(8, 3, 32, 32)

    model = Net()
    out1, out2, out3, out4 = model(input)

    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)