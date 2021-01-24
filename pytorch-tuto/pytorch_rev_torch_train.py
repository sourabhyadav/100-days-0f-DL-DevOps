# TORCH based training Automated

# Setps:
# 1) Design model (input, output size, forward pass)
# 2) Contruct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: compute gradients
#   - update weights: with optimizer  

import torch 
import torch.nn as nn

# Note: Modify the array size
X = torch.tensor([[1],[2],[3],[4]], dtype= torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype= torch.float32)

X_test = torch.tensor([5.0], dtype= torch.float32)

# w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# # model pred : Liner regression
# def forward(x):
#     return w * x

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

# Typically this is where the model architecture has to come
# model = nn.Linear(input_size, output_size) # basically a single layer model which is linear regression

# Create a cutom model for Linear Regression
class customLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(customLinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = customLinearRegression(input_size, output_size)

# loss : MSE
# def loss(y, y_pred):
#     return ((y_pred-y)**2).mean()

loss = nn.MSELoss()

# Not required now
# # compute gradients
# # dJ/dw = 1/N * 2x * (wx - y)
# def gradient(x, y, y_pred):
#     return np.dot(2*x, y_pred-y ).mean()

# print("Prediction before training: f(5) = ", forward(5))

print("Prediction before training: f(5) = ", model(X_test).item())


# Training
lr = 0.01
n_iters = 100

# optimizer = torch.optim.SGD([w], lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
    # forward pass
    # y_pred = forward(X)
    y_pred = model(X)

    # loss 
    l = loss(Y, y_pred) # it is torch callable function

    # gradient = backward pass 
    l.backward()    # Note: this will compute the gradients automatically

    # # update weights
    # with torch.no_grad():
    #     w -= lr * w.grad

    # Weights update will be done automatically
    optimizer.step()

    # zero grad. As by default gardients will automatically accumulate
    # w.grad.zero_()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        # print("epoch: ", epoch+1,  "w = ", w, "loss = ", l)
        print("epoch: ", epoch+1,  "w = ", w[0][0].item(), "loss = ", l)
    

# print("Prediction before training: f(5) = ", forward(5))

print("Prediction before training: f(5) = ", model(X_test).item())
