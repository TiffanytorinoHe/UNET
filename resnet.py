import torchvision
from torch import nn


def _resNetEncoder(resnet):
    """
    input size: 224*224*3
    output size: 14*14*512
    """
    resnet_features = list(resnet.children())[:-2]
    feature_layers = [nn.Sequential(*resnet_features[0:3]), nn.Sequential(*resnet_features[3:5])]
    feature_layers.extend(resnet_features[5:8])
    return feature_layers


def ResNet18Encoder(pretrained=False):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resNetEncoder(resnet)


def ResNet34Encoder(pretrained=False):
    resnet = torchvision.models.resnet34(pretrained=pretrained)
    return _resNetEncoder(resnet)
