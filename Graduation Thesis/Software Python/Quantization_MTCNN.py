import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant import Int32Bias
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU, QuantLinear
from collections import OrderedDict

class QuantFlatten(nn.Module):
    def __init__(self):
        super(QuantFlatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.reshape(x.size(0), -1)

class QuantPNet(nn.Module):
    def __init__(self, is_train=False):
        super(QuantPNet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('quant_inp', QuantIdentity(bit_width=4, return_quant_tensor=True)),
            ('conv1', QuantConv2d(3, 10, 3, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu1', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', QuantConv2d(10, 16, 3, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu2', QuantReLU(bit_width=4, return_quant_tensor=True)),

            ('conv3', QuantConv2d(16, 32, 3, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu3', QuantReLU(bit_width=4, return_quant_tensor=True))
        ]))

        self.conv4_1 = QuantConv2d(32, 2, 1, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)
        self.conv4_2 = QuantConv2d(32, 4, 1, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)

    def forward(self, x):
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)

        if not self.is_train:
            a = F.softmax(a, dim=1)
        return b, a

class QuantRNet(nn.Module):
    def __init__(self, is_train=False):
        super(QuantRNet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('quant_inp', QuantIdentity(bit_width=4, return_quant_tensor=True)),
            ('conv1', QuantConv2d(3, 28, 3, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu1', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', QuantConv2d(28, 48, 3, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu2', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', QuantConv2d(48, 64, 2, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu3', QuantReLU(bit_width=4, return_quant_tensor=True)),

            ('flatten', QuantFlatten()),
            ('conv4', QuantLinear(576, 128, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu4', QuantReLU(bit_width=4, return_quant_tensor=True))
        ]))

        self.conv5_1 = QuantLinear(128, 2, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)
        self.conv5_2 = QuantLinear(128, 4, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)

    def forward(self, x):
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)

        if not self.is_train:
            a = F.softmax(a, dim=1)
        return b, a

class QuantONet(nn.Module):
    def __init__(self, is_train=False):
        super(QuantONet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('quant_inp', QuantIdentity(bit_width=4, return_quant_tensor=True)),
            ('conv1', QuantConv2d(3, 32, 3, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu1', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', QuantConv2d(32, 64, 3, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu2', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', QuantConv2d(64, 64, 3, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu3', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', QuantConv2d(64, 128, 2, 1, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu4', QuantReLU(bit_width=4, return_quant_tensor=True)),

            ('flatten', QuantFlatten()),
            ('conv5', QuantLinear(1152, 256, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)),
            ('relu5', QuantReLU(bit_width=4, return_quant_tensor=True)),
        ]))

        self.conv6_1 = QuantLinear(256, 2, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)
        self.conv6_2 = QuantLinear(256, 4, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)
        self.conv6_3 = QuantLinear(256, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias, return_quant_tensor=True)

    def forward(self, x):
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)

        if not self.is_train:
            a = F.softmax(a, dim=1)
        return c, b, a