import torch 
import torch.nn as nn
import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Generate Logistic Regression dataset
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target
n_samples, n_features = X.shape 
print("num samples: ", n_samples, " num features: ", n_features)

# split our data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1)

# preprocessing our data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# convert to torch tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

# reshape
Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test = Y_test.view(Y_test.shape[0], 1)


class LogisticReg(nn.Module):
    
    def __init__(self, n_input_features):
        super(LogisticReg, self).__init__()
        self.lin = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.lin(x))
        return y_pred

model = LogisticReg(n_features)

lr = 0.01

# loss
creiterion = nn.BCELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr)

# training loop
num_epoces = 100

for epoch in range(num_epoces):
    # foreard pass
    y_pred = model(X_train)

    # compute loss
    loss = creiterion(y_pred, Y_train)

    # backward pass 
    loss.backward()

    # update the weights
    optimizer.step()

    # Reset the weithgts
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print("epoch: ", epoch+1, " loss: ", loss.item())


# Evaluate our model
with torch.no_grad():
    y_preds = model(X_test)
    y_pred_cls = y_preds.round()
    acc = y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print("Acc: ", acc)