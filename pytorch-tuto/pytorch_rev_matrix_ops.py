import torch 
import numpy as np

x = torch.rand(2,3)
y = torch.rand(2,3)

# element-wise add/sub/mul/div # all syantax are same
z = x + y
print(y)

z1 = torch.add(x,y)
print(z1)

# In place addition as any fucntion with _ is in-place stuff
y.add_(x)
print(y)

# array slicing
print("x: \n", x)
print(x[:, 0])

# getting element works 
print(x[1,2])           # will get only element with a tensor
print(x[1,2].item()) # this will get actual value of array

# reshaping : this is equivalant to reshape function in numpy
x = torch.rand(4,5)
print("x: ", x)
y = x.view(2,2,5)
print("y: ", y, " shape: ", y.shape)

# torch tensor to numpy array
a = x.numpy()
print("a: ", a)     # note: if the tensor and array are on CPU then they will be sharing memory
a[1,2] = 100
print("x: ", x)

# numpy array to tensor
n = np.ones(4)
print(n)
t = torch.from_numpy(n)
print(t)
n += 1
print(t)

# operations on GPU
if torch.cuda.is_available():   # Returns true if cuda is there
device = torch.device("cuda")           # selectes a default GPU
x = torch.ones(5, 5, device= device)    # creates a tensor on GPU
y = torch.ones(5)       # create a tensor on CPU
y = y.to(device)        # Transfer the CPU tensor to GPU
z = x+y                 # GPU operation
z_cpu = z.to("cpu")     # GPU to CPU transfer