from torch.nn.functional import cross_entropy
from torch.nn import NLLLoss
from torch.nn import LogSoftmax
from torch.nn import Softmax
from torch import nn
import torch

# x = torch.tensor([[0.1, 0.9]])
# print(LogSoftmax(x))
# y = torch.tensor([[1.0, 0.0]])
# print(cross_entropy(x, y))
# # print(NLLLoss(x, y))

# m = nn.LogSoftmax(dim=0)
# # m = nn.Softmax(dim=0)
# # input = torch.randn(2, 3)
# input = torch.tensor([[2.0], [3.0]])
# another_input = torch.tensor([[3.0], [4.0]])
# # print(input.shape)
# # output = m(input)
# # print(output)
# print(m(input))

# print(torch.log(torch.exp(torch.tensor([2])) / (torch.exp(torch.tensor([2])) + torch.exp(torch.tensor([3])))))

# print(torch.nn.NLLLoss(input, another_input))


# m = nn.LogSoftmax(dim=1)
# loss = nn.NLLLoss()
# # input is of size N x C = 3 x 5
# input = torch.randn(3, 5, requires_grad=True)
# input = torch.tensor([[1.0, 1.1, 1.3, 1.5, 2.0], [3.0, 3.1, 3.2, 3.5, 3.8], [5, 6, 3, 2, 1]], requires_grad=True)
# # each element in target has to have 0 <= value < C
# target = torch.tensor([1, 0, 4])

# output = loss(m(input), target)
# print(output)
# output.backward()


tensor1 = torch.tensor([[0.7, 0.2, 0.1], [0.8, 0.1, 0.1]], dtype=torch.float64)
# print(tensor1.shape)
tensor2 = torch.tensor([1, 2])
loss = torch.nn.NLLLoss()
print(loss(tensor1, tensor2))
