from torch import nn
import torch


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(20, 30)

    def forward(self, x):
        return self.fc1(x)

input = torch.rand(128, 20)
m = nn.Linear(20, 30)

custom_net = CustomNet()

print(custom_net(input))