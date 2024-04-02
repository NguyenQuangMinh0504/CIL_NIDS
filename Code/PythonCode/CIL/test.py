# from torch import Tensor
# import torch

# tensor = Tensor(
# [
#     [-3.6737,   3.6739,   7.9670,  -6.8142,   7.5660,  -7.4853],
#     [-3.8698,   3.8700,   8.3494,  -7.1354,   7.9662,  -7.8834],
#     [3.0694,  -3.0693,  -6.3453,   4.8575, -14.1109,  14.8372],
#     [3.0781,  -3.0780,  -6.3650,   4.8734, -14.1494,  14.8778],
#     [-3.7971,   3.7972,   8.2072,  -7.0160,   7.8165,  -7.7347],
#     [-3.7736,   3.7738,   8.1614,  -6.9775,   7.7684,  -7.6869],
#     [-3.7105,   3.7107,   8.0386,  -6.8743,   7.6404,  -7.5594],
#     [-3.7717,   3.7719,   8.1577,  -6.9744,   7.7645,  -7.6830],
#     [-3.8698,   3.8700,   8.3495,  -7.1355,   7.9662,  -7.8834],
#     [3.1876,  -3.1874,  -6.6086,   5.0743, -14.6059,  15.3588],
#     [-3.7871,   3.7873,   8.1883,  -7.0001,   7.7980,  -7.7159],
#     [3.1329,  -3.1327,  -6.4858,   4.9743, -14.3710,  15.1113],
#     [3.1295,  -3.1293,  -6.4794,   4.9677, -14.3636,  15.1034],
#     [-3.6428,   3.6429,   7.9025,  -6.7601,   7.4887,  -7.4108],
#     [3.0746,  -3.0745,  -6.3574,   4.8670, -14.1357,  14.8634],
#     [3.2236,  -3.2235,  -6.6936,   5.1387, -14.7841,  15.5461],
#     [3.1713,  -3.1712,  -6.5750,   5.0436, -14.5523,  15.3020],
#     [-3.7287,   3.7289,   8.0736,  -6.9037,   7.6759,  -7.5950],
#     [3.0622,  -3.0621,  -6.3294,   4.8443, -14.0822,  14.8070],
#     [3.2201,  -3.2200,  -6.6833,   5.1333, -14.7547,  15.5154],
#     [-3.4313,   3.4315,   7.4917,  -6.4150,   7.0624,  -6.9858],
#     [-3.6644,   3.6645,   7.9488,  -6.7989,   7.5468,  -7.4662]
# ]
# )
# target = Tensor(
#     [4, 4, 5, 5, 4, 4, 4, 4, 4, 5, 4, 5, 5, 4, 5, 5, 5, 4, 5, 5, 4, 4]
# )
# print(target)
# prediction = torch.max(tensor, dim=1)
# print(prediction)
# target = target.to(dtype=torch.int64)
# fake_target = target - 4
# print(F.cross_entropy(tensor[:, 4:], fake_target))

# from torch.nn import LogSoftmax, NLLLoss, Softmax
# print(F.cross_entropy(Tensor([[7.5660,  -7.4853]]), Tensor([0]).to(torch.int64)))
# log_softmax = LogSoftmax(dim=1)
# softmax = Softmax(dim=1)
# print(log_softmax(Tensor([[7.5660,  -7.4853]])))
# print(softmax(Tensor([[7.5660,  -7.4853]])))
# loss = NLLLoss()
# print(loss(log_softmax(Tensor([[7.5660,  -7.4853]])), Tensor([0]).to(torch.int64)))

import torch
from torch import tensor
from torch.nn import functional as F

# Train on class 4-5
outputs = tensor([[-3.6737,   3.6739,   7.9670,  -6.8142,   7.5660,  -7.4853]])
# Predict on class 0-5
predicts = torch.max(outputs, dim=1)
print(predicts.indices)  # Result is 2
target = tensor([4], dtype=torch.int64)  # Target is 4
fake_target = target - 4  # Fake target is 0
loss = F.cross_entropy(outputs[:, 4:], fake_target)
print(round(loss.item(), ndigits=3))  # Loss is 0
