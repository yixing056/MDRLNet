import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.pvtv2 import pvt_v2_b2


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


#解码器，通过上采样和卷积操作，从高层特征图和低层特征图中提取信息，生成新的特征图，以逐步恢复空间分辨率。
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2) #上采样层
        in_channels = in_channels_high + in_channels_low   #解码层
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, high_feat, low_feat):
        x = self.up(high_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


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
        path = r'checkpoints/weight/pvtv2_BGNet-59.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.decoder3 = DecoderBlock(512, 320, 320)
        self.decoder2 = DecoderBlock(320, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        self.predictor = nn.Conv2d(64, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)  # [64, 128, 320, 512]
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)

        # Decoder
        x34 = self.decoder3(x4, x3)
        x234 = self.decoder2(x34, x2)
        x1234 = self.decoder1(x234, x1)

        out = self.predictor(x1234)
        out= F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out,self.sigmoid(out)


if __name__ == '__main__':
    input = torch.rand(8, 3,352, 352)

    model = Net()
    out= model(input)

    print(out.shape)

