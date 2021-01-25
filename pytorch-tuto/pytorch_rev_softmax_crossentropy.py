import torch 
import torch.nn as nn 
import numpy as np 


# Normally softmax is calculated to get the weights b/w 0-1 with 
# higer values to higher logits

# Cross-entropy loas is caluclated with On-Hot-Encoded values

# However, nn.CrossEntropyLoass -> nn.LogSoftman + nn.NLLLoss
# Thus, for Y_gt has class labels, not One-hot
# Y_pred has raw scores (logits) no softmax

loss = nn.CrossEntropyLoss()

Y = torch.tensor([2,0,1])

# n_samples x n_classes = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1], [2.0, 1.2, 0.1], [2.0, 2.0, 1.1]])
Y_pred_bad = torch.tensor([[1.0, 2.0, 0.1], [0.0, 1.3, 0.1], [2.2, 1.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print("l1: ", l1.item())
print("l2: ", l2.item())

_, pred1 = torch.max(Y_pred_good, 1)
_, pred2 = torch.max(Y_pred_bad, 1)

print("pred1: ", pred1)
print("pred2: ", pred2)