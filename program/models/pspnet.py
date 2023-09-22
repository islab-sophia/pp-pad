import torch
import torch.nn as nn
import torch.nn.functional as F

import models.pppad as pppad
import models.partialpad as partialpad
import models.cap as cap

PARTIAL_PADDING_MODE = 'zeros' # 'zeros', 'reflect', 'replicate', 'circular' for base padding method in partial padding
CAP_PADDING_MODE = 'zeros' # 'zeros', 'reflect', 'replicate', 'circular' for base padding method in CAP

class PSPNet(nn.Module):
    def __init__(self, n_classes, img_size, padding_mode='pp-pad'):
        super(PSPNet, self).__init__()

        # paramter settings
        self.padding_mode = padding_mode
        block_config = [3, 4, 6, 3]  # resnet50
        # if padding_mode in ('cap_pretrain', 'cap_train'):
        #     img_size = 512
        # else:
        #     img_size = 475

        # Subnetworks (four modules)
        self.feature_conv = FeatureMap_convolution(padding_mode)
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1, padding_mode=padding_mode)
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1, padding_mode=padding_mode)
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2, padding_mode=padding_mode)
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4, padding_mode=padding_mode)

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[
            6, 3, 2, 1], padding_mode=padding_mode)

        self.decode_feature = DecodePSPFeature(
            height=img_size, width=img_size, n_classes=n_classes, padding_mode=padding_mode)

        self.aux = AuxiliaryPSPlayers(
            in_channels=1024, height=img_size, width=img_size, n_classes=n_classes, padding_mode=padding_mode)

    def forward(self, x):
        if self.padding_mode == 'cap_pretrain':
            [x, cap_loss] = self.feature_conv(x)
        else:
            x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)

        output_aux = self.aux(x)  # intermediate fatures to aux module

        x = self.feature_dilated_res_2(x)

        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)
        if self.padding_mode == 'cap_pretrain':
            return (output, output_aux, cap_loss)
        else:
            return (output, output_aux)


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, padding_mode='zeros'):
        super(conv2DBatchNormRelu, self).__init__()
        self.padding_mode = padding_mode
        if padding_mode == 'pp-pad':
            self.calc_padding = pppad.calc_padding(padding, in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding=0, dilation=dilation, bias=bias)
        elif padding_mode == 'partial':
            self.conv = partialpad.PartialConv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias, padding_mode=PARTIAL_PADDING_MODE)
        elif padding_mode in ('cap_pretrain', 'cap_train'):
            self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias, padding_mode=CAP_PADDING_MODE)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias, padding_mode=padding_mode)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        # Supressing memory usage by calculating outputs with inplace setting without saving inputs
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.padding_mode == 'pp-pad':
            x = self.calc_padding(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs

class FeatureMap_convolution(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super(FeatureMap_convolution, self).__init__()

        self.padding_mode = padding_mode

        # Conv1
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        if padding_mode in ('cap_pretrain', 'cap_train'):
            self.calc_padding_CAP = cap.calc_padding_CAP(padding, padding_mode=padding_mode)
        self.cbnr_1 = conv2DBatchNormRelu(
                in_channels, out_channels, kernel_size, stride, padding, dilation, bias, padding_mode=padding_mode)

        # Conv2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(
                in_channels, out_channels, kernel_size, stride, padding, dilation, bias, padding_mode=padding_mode)

        # Conv3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(
                in_channels, out_channels, kernel_size, stride, padding, dilation, bias, padding_mode=padding_mode)

        # MaxPooling
        if padding_mode == 'pp-pad':
            self.calc_padding_pooling = pppad.calc_padding(padding=1, in_channels=out_channels)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.padding_mode == 'cap_pretrain':
            [_, cap_loss] = self.calc_padding_CAP(x)
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        if self.padding_mode == 'pp-pad':
            x = self.calc_padding_pooling(x)
        outputs = self.maxpool(x)
        if self.padding_mode == 'cap_pretrain':
            return [outputs, cap_loss]
        else:
            return outputs

class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation, padding_mode='zeros'):
        super(ResidualBlockPSP, self).__init__()

        # bottleNeckPSP
        self.add_module(
            "block1",
            bottleNeckPSP(in_channels, mid_channels,
                          out_channels, stride, dilation, padding_mode=padding_mode)
        )

        # Repeating bottleNeckIdentifyPSP
        for i in range(n_blocks - 1):
            self.add_module(
                "block" + str(i+2),
                bottleNeckIdentifyPSP(
                    out_channels, mid_channels, stride, dilation, padding_mode=padding_mode)
            )


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, padding_mode='zeros'):
        super(conv2DBatchNorm, self).__init__()
        self.padding_mode = padding_mode
        if padding_mode == 'pp-pad':
            self.calc_padding = pppad.calc_padding(padding, in_channels=in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding=0, dilation=dilation, bias=bias)
        elif padding_mode == 'partial':
            self.conv = partialpad.PartialConv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias, padding_mode=PARTIAL_PADDING_MODE)
        elif padding_mode in ('cap_pretrain', 'cap_train'):
            self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias, padding_mode=CAP_PADDING_MODE)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias, padding_mode=padding_mode)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.padding_mode == 'pp-pad':
            x = self.calc_padding(x)
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation, padding_mode='zeros'):
        super(bottleNeckPSP, self).__init__()

        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, padding_mode=padding_mode)
        self.cbr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False, padding_mode=padding_mode)
        self.cb_3 = conv2DBatchNorm(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, padding_mode=padding_mode)

        # Skip connection
        self.cb_residual = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False, padding_mode=padding_mode)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        return self.relu(x_conv + residual)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation, padding_mode='zeros'):
        super(bottleNeckIdentifyPSP, self).__init__()

        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, padding_mode=padding_mode)
        self.cbr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False, padding_mode=padding_mode)
        self.cb_3 = conv2DBatchNorm(
            mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, padding_mode='zeros'):
        super(PyramidPooling, self).__init__()

        # Output channels in each conv layer
        out_channels = int(in_channels / len(pool_sizes))

        # Conv Layers
        # pool_sizes: [6, 3, 2, 1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, padding_mode=padding_mode)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, padding_mode=padding_mode)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, padding_mode=padding_mode)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, padding_mode=padding_mode)

    def forward(self, x):
        height = x.shape[2]
        width = x.shape[3]

        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(
            height, width), mode="bilinear", align_corners=True)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(
            height, width), mode="bilinear", align_corners=True)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(
            height, width), mode="bilinear", align_corners=True)

        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(
            height, width), mode="bilinear", align_corners=True)

        # Concat features along dim=1
        output = torch.cat([x, out1, out2, out3, out4], dim=1)

        return output

class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes, padding_mode='zeros'):
        super(DecodePSPFeature, self).__init__()

        # Image size
        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(
            in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, padding_mode=padding_mode)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output


class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes, padding_mode='zeros'):
        super(AuxiliaryPSPlayers, self).__init__()

        # Image size
        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(
            in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, padding_mode=padding_mode)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output


