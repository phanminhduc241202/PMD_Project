import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int32Bias
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU, QuantLinear
from collections import OrderedDict

class QuantFlatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.reshape(x.size(0), -1)

# ---------------------- PNet ----------------------
class QuantPNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input quantization
        self.quant_inp = QuantIdentity(bit_width=4, return_quant_tensor=True)
        # feature layers
        self.features = nn.Sequential(OrderedDict([
            ('conv1', QuantConv2d(3, 10, 3, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu1', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),
            ('conv2', QuantConv2d(10, 16, 3, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu2', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('conv3', QuantConv2d(16, 32, 3, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu3', QuantReLU(bit_width=4, return_quant_tensor=True)),
        ]))
        # output heads
        self.bbox_head = QuantConv2d(32, 4, 1, 1, bias=True,
                                     weight_bit_width=4, bias_quant=Int32Bias,
                                     return_quant_tensor=True)
        self.conf_head = QuantConv2d(32, 2, 1, 1, bias=True,
                                     weight_bit_width=4, bias_quant=Int32Bias,
                                     return_quant_tensor=True)
        # force-output quantization identities (INT4)
        self.quant_bbox = QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.quant_conf = QuantIdentity(bit_width=4, return_quant_tensor=True)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        bbox = self.bbox_head(x)
        conf = self.conf_head(x)
        bbox = self.quant_bbox(bbox)
        conf = self.quant_conf(conf)
        # trả về QTensor INT4, không có softmax
        return bbox, conf

# ---------------------- RNet ----------------------
class QuantRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_inp = QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', QuantConv2d(3, 28, 3, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu1', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv2', QuantConv2d(28, 48, 3, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu2', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv3', QuantConv2d(48, 64, 2, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu3', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('flatten', QuantFlatten()),
            ('fc1', QuantLinear(576, 128, bias=True,
                                weight_bit_width=4, bias_quant=Int32Bias,
                                return_quant_tensor=True)),
            ('relu4', QuantReLU(bit_width=4, return_quant_tensor=True)),
        ]))
        self.bbox_head = QuantLinear(128, 4, bias=True,
                                     weight_bit_width=4, bias_quant=Int32Bias,
                                     return_quant_tensor=True)
        self.conf_head = QuantLinear(128, 2, bias=True,
                                     weight_bit_width=4, bias_quant=Int32Bias,
                                     return_quant_tensor=True)
        self.quant_bbox = QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.quant_conf = QuantIdentity(bit_width=4, return_quant_tensor=True)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        bbox = self.bbox_head(x)
        conf = self.conf_head(x)
        bbox = self.quant_bbox(bbox)
        conf = self.quant_conf(conf)
        return bbox, conf

# ---------------------- ONet ----------------------
class QuantONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_inp = QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', QuantConv2d(3, 32, 3, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu1', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv2', QuantConv2d(32, 64, 3, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu2', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv3', QuantConv2d(64, 64, 3, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu3', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),
            ('conv4', QuantConv2d(64, 128, 2, 1, bias=True,
                                  weight_bit_width=4, bias_quant=Int32Bias,
                                  return_quant_tensor=True)),
            ('relu4', QuantReLU(bit_width=4, return_quant_tensor=True)),
            ('flatten', QuantFlatten()),
            ('fc1', QuantLinear(1152, 256, bias=True,
                                weight_bit_width=4, bias_quant=Int32Bias,
                                return_quant_tensor=True)),
            ('relu5', QuantReLU(bit_width=4, return_quant_tensor=True)),
        ]))
        self.bbox_head = QuantLinear(256, 4, bias=True,
                                     weight_bit_width=4, bias_quant=Int32Bias,
                                     return_quant_tensor=True)
        self.conf_head = QuantLinear(256, 2, bias=True,
                                     weight_bit_width=4, bias_quant=Int32Bias,
                                     return_quant_tensor=True)
        # ví dụ thêm head cho landmark (5 điểm × 2D = 10)
        self.landmark_head = QuantLinear(256, 10, bias=True,
                                         weight_bit_width=4, bias_quant=Int32Bias,
                                         return_quant_tensor=True)
        self.quant_bbox     = QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.quant_conf     = QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.quant_landmark = QuantIdentity(bit_width=4, return_quant_tensor=True)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        bbox      = self.bbox_head(x)
        conf      = self.conf_head(x)
        landmark  = self.landmark_head(x)
        bbox      = self.quant_bbox(bbox)
        conf      = self.quant_conf(conf)
        landmark  = self.quant_landmark(landmark)
        return bbox, conf, landmark
