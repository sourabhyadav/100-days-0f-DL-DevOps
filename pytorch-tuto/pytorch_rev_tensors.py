import torch 

# Note: torch arrays are almost similar to numpy arrays

# Torch array initilizing using methods

# empty: creates empty array with some garbage values
x = torch.empty(3, 2, 5)
print("x: \n", x, " x.shape: ", x.shape)

# rand
x = torch.rand(3, 2, 5)
print("x: \n", x, " x.shape: ", x.shape)

# zeros
x = torch.zeros(3, 2, 5)
print("x: \n", x, " x.shape: ", x.shape)

# ones
x = torch.ones(3, 2, 5)
print("x: \n", x, " x.shape: ", x.shape)

# ones
x = torch.ones(3, 2, 5, dtype= torch.int)
print("x: \n", x, " x.shape: ", x.shape)

# python list to torch tensor
x = torch.tensor([2.4, 19.6, 6])
print("x: \n", x, " x.shape: ", x.shape)

# python list to torch tensor
x = torch.tensor([ [2.4, 19.6, 6], [0.2, 0.5, 0.3] ])
print("x: \n", x, " x.shape: ", x.shape)