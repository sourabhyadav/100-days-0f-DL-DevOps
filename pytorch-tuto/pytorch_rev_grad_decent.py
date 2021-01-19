'''
# NUMPY based manually training. Later will make it completely with torch

import numpy as np

X = np.array([1,2,3,4], dtype= np.float32)
Y = np.array([2,4,6,8], dtype= np.float32)

w = 0.0

# model pred : Liner regression
def forward(x):
    return w * x

# loss : MSE
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

# compute gradients
# dJ/dw = 1/N * 2x * (wx - y)
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y ).mean()

print("Prediction before training: f(5) = ", forward(5))

# Training
lr = 0.01
n_iters = 10

for epoch in range(n_iters):
    # forward pass
    y_pred = forward(X)

    # loss 
    l = loss(Y, y_pred)

    # gradient 
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= lr * dw

    if epoch % 1 == 0:
        print("epoch: ", epoch+1,  "w = ", w, "loss = ", l)
    

print("Prediction after training: f(5) = ", forward(5))

'''

'''
# TORCH based training 1
import torch 

X = torch.tensor([1,2,3,4], dtype= torch.float32)
Y = torch.tensor([2,4,6,8], dtype= torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# model pred : Liner regression
def forward(x):
    return w * x

# loss : MSE
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

# Not required now
# # compute gradients
# # dJ/dw = 1/N * 2x * (wx - y)
# def gradient(x, y, y_pred):
#     return np.dot(2*x, y_pred-y ).mean()

print("Prediction before training: f(5) = ", forward(5))

# Training
lr = 0.01
n_iters = 100

for epoch in range(n_iters):
    # forward pass
    y_pred = forward(X)

    # loss 
    l = loss(Y, y_pred)

    # gradient = backward pass 
    l.backward()    # Note: this will compute the gradients automatically

    # update weights
    with torch.no_grad():
        w -= lr * w.grad

    # zero grad. As by default gardients will automatically accumulate
    w.grad.zero_()

    if epoch % 10 == 0:
        print("epoch: ", epoch+1,  "w = ", w, "loss = ", l)
    

print("Prediction after training: f(5) = ", forward(5))
'''

