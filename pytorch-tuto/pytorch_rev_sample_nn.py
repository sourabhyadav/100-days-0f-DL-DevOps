# For Binary classification --> nn -- logits -- sigmoid -- BCE loss
# For multi-class classification --> nn -- logits -- softmax -- CrossEntropyLoss

import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper params
input_size = 784 # 28x28
hidden_size = 100
num_class = 10
batch_size = 100
num_epochs = 10
lr = 0.001

# load dataset MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',
                train= True,
                transform=transforms.ToTensor(),
                download= True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                train= False,
                transform=transforms.ToTensor(),
                download= True)

# Pass this datasets class to dataloader to use in-built functions 
train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=False)

# Print the dataset
examples = iter(train_loader)
samples, labels = examples.next()
print("samples shape: ", samples.shape, " labels.shape: ", labels.shape)

# Display the dataset
for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
plt.show()

# implement FC Network for classificaiton
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # here we dont apply softmax as softmaz will be part of CrossEntropy loss of pytorch
        return out

# create our model object
model = NeuralNet(input_size, hidden_size, num_class)

# loss 
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


n_total_steps = len(train_loader)

# training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # original shape: 100, 1, 28, 28
        # input layer size: 100, 784
        # batch_size = 100
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward
        pred = model(images)

        # compute loss
        loss = criterion(pred, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # update the prarms
        optimizer.step()

        # print the progress
        if (i+1) % 100 == 0:
            print("epoch: (", epoch+1, "/", num_epochs, ") step: (", i+1, "/", n_total_steps, ") loss: ", loss.item())

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        # prepare data
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # inference
        pred = model(images)
        # value, index
        _, preds = torch.max(pred, 1)
        
        # calculate accuracy
        n_samples = labels.shape[0]
        n_correct += (preds == labels).sum().item()
    
    acc = 100.0 * n_correct / n_samples
    print("Accuracy: ", acc)