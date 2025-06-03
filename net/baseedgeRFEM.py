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
        self.reduce1 = Conv1x1(64, 64)  #1x1 卷积层，用于减少特征图的通道数。
        #self.reduce4 = Conv1x1(2048, 256)
        self.reduce4 = Conv1x1(512, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:] #获取尺寸信息
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)  #将x4插值到与x1相同的空间尺寸，双线性插值方法
        out = torch.cat((x4, x1), dim=1) #在通道维度为1上，将x4与x1拼接起来，得到新的特征图
        out = self.block(out)

        return out


class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)  #定义一个1D卷积层，用于处理池化后的特征。
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)  #检查输入特征图c和注意力图att的尺寸是否相同。如果不同，则使用双线性插值方法将att调整到与c相同的空间尺寸。
        x = c * att + c  #注意力加权操作，增强了特征图中的显著部分。
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        # a = wei.squeeze(-1)
        # b = a.transpose(-1, -2)
        # c =  self.conv1d(b).transpose(-1, -2)
        # d = c.unsqueeze(-1)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  #池化后的特征图进行1D卷积，生成通道注意力权重。
        wei = self.sigmoid(wei)   #使用Sigmoid激活函数生成通道注意力权重，将其值缩放到0到1之间。
        x = x * wei  #将卷积后的特征图x与注意力权重逐元素相乘，进行通道加权操作。

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
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)  #检查低分辨率特征图lf和高分辨率特征图hf的空间尺寸是否相同。如果不同，则使用双线性插值方法将hf调整到与lf相同的空间尺寸。
        x = torch.cat((lf, hf), dim=1)  #将低分辨率特征图lf和高分辨率特征图hf在通道维度上拼接起来。
        x=self.conv1_1(x)
        r1 = self.dconv_r2(x)
        r1 = torch.cat([r1, x], dim=1)

        r2 = self.dconv_r4(self.conv1_1x1(r1))
        r2 = torch.cat([r2, r1], dim=1)

        r3 = self.dconv_r8(self.conv2_1x1(r2))
        r3 = torch.cat([r3, r2], dim=1)

        out = self.conv3_1x1(r3)

        return out

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
        path = r'D:\SOD\MyNet_BGNet\models\pvt_v2_b2.pth'
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

        # self.reduce1 = Conv1x1(64, 64)
        # self.reduce2 = Conv1x1(128, 128)
        # self.reduce3 = Conv1x1(320, 256)
        # self.reduce4 = Conv1x1(512, 256)

        self.rfem1 = RFEM(128, 64, 64)
        self.rfem2 = RFEM(320, 128, 128)
        self.rfem3 = RFEM(512, 320, 320)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        # self.predictor2 = nn.Conv2d(128, 1, 1)
        # self.predictor3 = nn.Conv2d(256, 1, 1)

        self.sigmoid = nn.Sigmoid()


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

        # x1r = self.reduce1(x1a)
        # x2r = self.reduce2(x2a)
        # x3r = self.reduce3(x3a)
        # x4r = self.reduce4(x4a)

        x34 = self.rfem3(x3a, x4a)
        x234 = self.rfem2(x2a, x34)
        x1234 = self.rfem1(x1a, x234)

        # o3 = self.predictor3(x34)
        # o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        # o2 = self.predictor2(x234)
        # o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        # o1 = self.predictor1(x1234)
        # o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)
        out = self.predictor1(x1234)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out, self.sigmoid(out),oe

        # return o3, o2, o1, oe


if __name__ == '__main__':
    input = torch.rand(8, 3, 352, 352)

    model = Net()
    out1, out2, out3, out4 = model(input)

    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)