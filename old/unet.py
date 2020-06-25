import torch
from vgg import *
from resnet import *


__all__ = ['UnetResNet18', 'UnetResNet34', 'UnetVGG13', 'UnetVGG16']


class DecodeLayer(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch+mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input1, input2):
        x = self.up(input1)
        x = torch.cat((x, input2), dim=1)
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, out_ch, encoder, num_channels, freeze_encoder):
        super().__init__()
        self.encoder1 = encoder[0]
        self.encoder2 = encoder[1]
        self.encoder3 = encoder[2]
        self.encoder4 = encoder[3]
        self.encoder5 = encoder[4]

        # 这里的代码冻结encoder参数，使用预训练模型时生效
        if freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False

        self.decoder1 = DecodeLayer(num_channels[4], num_channels[3], num_channels[3])
        self.decoder2 = DecodeLayer(num_channels[3], num_channels[2], num_channels[2])
        self.decoder3 = DecodeLayer(num_channels[2], num_channels[1], num_channels[1])
        self.decoder4 = DecodeLayer(num_channels[1], num_channels[0], num_channels[0])
        self.classifier = nn.Conv2d(num_channels[0], out_ch, 1)

    def forward(self, input):
        e1 = self.encoder1(input)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d = self.decoder1(e5, e4)
        d = self.decoder2(d, e3)
        d = self.decoder3(d, e2)
        d = self.decoder4(d, e1)

        output = self.classifier(d)
        # output = nn.Sigmoid()(output)

        return output


def UnetVGG13(out_ch, pretrained=False, freeze_encoder=False):
    num_channels = [64, 128, 256, 512, 512]  # vgg的channels
    encoder = VGG13Encoder(pretrained)
    return Unet(out_ch, encoder, num_channels, freeze_encoder)


def UnetVGG16(out_ch, pretrained=False, freeze_encoder=False):
    num_channels = [64, 128, 256, 512, 512]  # vgg的channels
    encoder = VGG16Encoder(pretrained)
    return Unet(out_ch, encoder, num_channels, freeze_encoder)


def UnetResNet18(out_ch, pretrained=False, freeze_encoder=False):
    num_channels = [64, 64, 128, 256, 512]  # resnet的channels
    encoder = ResNet18Encoder(pretrained)
    return Unet(out_ch, encoder, num_channels, freeze_encoder)


def UnetResNet34(out_ch, pretrained=False, freeze_encoder=False):
    num_channels = [64, 64, 128, 256, 512]  # resnet的channels
    encoder = ResNet34Encoder(pretrained)
    return Unet(out_ch, encoder, num_channels, freeze_encoder)
