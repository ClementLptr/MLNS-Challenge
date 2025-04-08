import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

a = torch.tensor([[1, 2, 3, 4, 4, 5], [2, 3, 45, 4, 1, 3], [2, 3, 4, 1, 9, 4]])

ones_indices = torch.tensor([[0, 1], [1, 2], [2, 3]])

a[:, ones_indices] = 1

print(a)
