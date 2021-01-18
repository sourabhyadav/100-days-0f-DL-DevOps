import torch 

x = torch.tensor(1.0)   # input vector/scalar
y = troch.tensor(2.0)   # output or ground truth

w = torch.tensor(1.0, requires_grad=True)   # initial weights

# Apply simple training

# step 1: forward pass and compute loss
y_hat = w * x           # compute the loss
loss = (y_hat - y)**2    # loss function

print("loss: \n", loss)

# step 2: create local grads or partial gradients 
# this will be done automatically by pytorch

# step 3: backward pass: Apply chain rule
loss.backward()
print("w.grad: ", w.grad)   

# step 4: Update weights and do this for a few iterations and epochs
# Basically apply Gradient Decent Process with an optimizer

