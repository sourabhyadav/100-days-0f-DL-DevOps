import torch 
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 

# Generate Regression dataset
X_numpy, Y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)

# convert to torch tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

# reshape
Y - Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape 

# model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

lr = 0.01

# loss
creiterion = nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr)

# training loop
num_epoces = 100

for epoch in range(num_epoces):
    # foreard pass
    y_pred = model(X)

    # compute loss
    loss = creiterion(y_pred, Y)

    # backward pass 
    loss.backward()

    # update the weights
    optimizer.step()

    # Reset the weithgts
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print("epoch: ", epoch+1, " loss: ", loss.item())


# plot the prediction

# stop calculating gradients
preds = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, preds, 'b')
plt.show()