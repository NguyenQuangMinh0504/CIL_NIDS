from torch import nn
import torch
depth = 32
layer_blocks = (depth - 2) // 6  # layer_blocks = 5
inplanes = 16
planes = 16
stride = 1


conv_a = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
bn_a = nn.BatchNorm2d(num_features=planes)
conv_b = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
bn_b = nn.BatchNorm2d(num_features=planes)
model = nn.Sequential(conv_a, bn_a, conv_b, bn_b)
print(model)

# Pass a dummy input tensor to the model
dummy_input = torch.randn(4, 16, 16, 16)

output = model(dummy_input)
print(output.size())