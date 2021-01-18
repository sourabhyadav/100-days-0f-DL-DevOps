# For any DL we will require Gradient calculation of tensors.
# Autograd package does all the heavy lifting and makes our life easy

import torch 
import numpy as np 

x = torch.randn(3, requires_grad = True)
print("x: ", x)
y = x + 2
print("y: \n", y)   # has gardient 
z = y*y*3
print("z: \n", z)   # has gardient 

# get the backward pass 
z.backward(x)                    # Computes the backward pass dx/dx
print("x.grad: ", x.grad)     # grad is available only for scalar values

# Stopping the gradient function for any tensor
# there are 3 ways to do this

# tensor.requires_grad_(False)
#x.requires_grad_(False)    # Note: _ functions are inplace stuff
#print("y: \n", x)       # observe that grad_fn is gone. So pytorch does not track stuff now.

# detach()
d = x.detach()
print("d \n", d)

# with torch.no_grad()
with torch.no_grad():
    k = x + 2
    print("k \n", k)           # note that above y = x+2 has grad_fn but not for k 


# gradients will be accumulated
weights = torch.ones(4, requires_grad= True)

for epoch in range(3):
    model_out = (weights*3).sum()

    model_out.backward()

    print("weights grad: \n", weights.grad)     # Note: these grads are accumulated for each epoc. Thus we need to reset those values while training
    
    # reset the grad to stop getting accumulated
    weights.grad.zero_()
