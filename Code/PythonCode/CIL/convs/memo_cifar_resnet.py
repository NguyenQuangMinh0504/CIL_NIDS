import math
import torch
from torch import nn
import torch.nn.functional as F
import logging
from typing import Type
from torch import Tensor


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x: Tensor):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResnetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super(ResnetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(in_channels=inplanes, out_channels=planes,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(num_features=planes)

        self.conv_b = nn.Conv2d(in_channels=planes, out_channels=planes,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(num_features=planes)
        self.downsample = downsample
        # logging.info(
        #     f"Resnet Basic block input: inplanes: {inplanes}; planes: {planes}, strides: {stride}, downsample: {downsample}"
        #     )

    def forward(self, x: Tensor):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + basicblock, inplace=True)


class GeneralizedResNet_cifar(nn.Module):
    """Include 2 stages, each stage includes 4 basic resnet block"""
    def __init__(self, block, depth, channels=3):
        super(GeneralizedResNet_cifar, self).__init__()
        assert (depth - 2) % 6 == 0, "depth should be one of 20, 32, 44, 56, 110"
        layer_blocks = (depth - 2) // 6

        # logging.info("Generalized Resnet Cifar")

        self.conv_1_3x3 = nn.Conv2d(in_channels=channels, out_channels=16,
                                    kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16

        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)

        # logging.info(f"Structure of stage 1 is: {self.stage_1}")

        self.stage_2 = self._make_layer(block=block, planes=32, blocks=layer_blocks, stride=2)

        # logging.info(f"Structure of stage 2 is: {self.stage_2}")

        self.out_dim = 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block: Type[ResnetBasicblock], planes, blocks: int, stride=1):
        """
        Args:
        blocks: Number of basic blocks in a layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(nIn=self.inplanes, nOut=planes * block.expansion, stride=stride)
        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        logging.info(f"Size of x_1 is: {x_1.size()}")
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        return x_2


class SpecializedResNet_cifar(nn.Module):
    def __init__(self, block, depth, inplanes=32, feature_dim=64):
        super(SpecializedResNet_cifar, self).__init__()
        self.inplanes = inplanes
        self.feature_dim = feature_dim
        layers_block = (depth - 2) // 6
        self.final_stage = self._make_layer(block=block, planes=64, blocks=layers_block, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block: Type[ResnetBasicblock], planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(nIn=self.inplanes, nOut=planes * block.expansion, stride=stride)
        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))
        return nn.Sequential(*layers)


def get_resnet32_a2fc() -> (nn.Module, nn.Module):
    basenet = GeneralizedResNet_cifar(block=ResnetBasicblock, depth=32)
    adaptivenet = SpecializedResNet_cifar(block=ResnetBasicblock, depth=32)
    return basenet, adaptivenet
