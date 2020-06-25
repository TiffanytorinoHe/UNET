import torchvision
from torch import nn

'''
class VGGEncoder(nn.Module):
    """
    input size: 224*224*3
    output size: 14*14*512
    """
    def __init__(self, feature_layers):
        super().__init__()
        self.feature_layers = feature_layers

    def forward(self, input):
        features = []
        x = input
        for layer in self.feature_layers:
            x = layer(x)
            features.append(x)
        return tuple(features)
'''


def VGG13Encoder(pretrained=False):
    vgg = torchvision.models.vgg13_bn(pretrained=pretrained)
    vgg_features = list(vgg.children())[0]
    feature_layers = [vgg_features[0:7], vgg_features[7:14], vgg_features[14:21], vgg_features[21:28], vgg_features[28:35]]
    return feature_layers


def VGG16Encoder(pretrained=False):
    vgg = torchvision.models.vgg16_bn(pretrained=pretrained)
    vgg_features = list(vgg.children())[0]
    feature_layers = [vgg_features[0:7], vgg_features[7:14], vgg_features[14:24], vgg_features[24:34], vgg_features[34:44]]
    return feature_layers
