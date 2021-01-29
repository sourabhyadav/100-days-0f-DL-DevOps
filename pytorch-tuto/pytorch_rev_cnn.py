import torch 
import torch.nn as nn 
import torch.nn.functional as f 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib as plt
import numpy as np 

# provide the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# some hyper params
num_epochs = 10
batch_size = 4
lr         = 0.001

# trasform PILImage(0,1) to normalized tensors (-1,1)
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# download the CFIAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                download= True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                download= True, transform=transform)

# wrap around the dataset with dataloader 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
'dog', 'frog', 'horse', 'ship', 'truck')

num_classes = 10

# Genereate the model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Just define all the layers
        self.conv1  = nn.Conv2d(3, 6, 5)    # inp_ch = 3, out_ch = 6, kernel_size = 5x5
        self.pool   = nn.MaxPool2d(2, 2)    # kernel_size = 2, stride = 2
        self.conv2  = nn.Conv2d(6, 16, 5)
        self.fc1    = nn.Linear(16*5*5, 120)
        self.fc2    = nn.Linear(120, 84)
        self.fc3    = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # connect the layers and perform forward pass
        x = self.pool(f.relu(self.conv1(x)))    # conv layer 1
        x = self.pool(f.relu(self.conv2(x)))    # conv layer 2
        x = x.view(-1, 16*5*5)                  # flatten before FC
        x = f.relu(self.fc1(x))                 # fc layer 1
        x = f.relu(self.fc2(x))                 # fc layer 2
        x = self.fc3(x)                         # No activation fucntion here as the softmax is applied inside the CrossEntropyLoss function
        return x

# load the model
model = ConvNet().to(device)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# training loop
n_total_steps = len(train_loader)
print("steps: ", n_total_steps)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # since we have used the batch_size = 4, the shape is [4,3,32,32] = 4,3,1024
        # load the batch to device
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)

        # compute loss
        loss = criterion(outputs, labels)

        # backward pass compute grad 
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        # print training gprgress
        if (i+1)% 2000 == 0:
            print("Epoch: ", epoch+1, "/", num_epochs, " setp: ", i+1, "/", n_total_steps, " loss: ", loss.item())

print("Training completed")

# Testing the model Accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (preds == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred  = preds[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print("Network Accuracy: ", acc)

    # accuracy per class
    for i in range(num_classes):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print("Accuracy for: ", classes[i], " acc: ", acc)