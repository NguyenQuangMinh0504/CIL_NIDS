import torch
a = torch.Tensor([[1, 2], [3, 4]])
print(torch.topk(a, k=2, dim=1, largest=True, sorted=True)[1])