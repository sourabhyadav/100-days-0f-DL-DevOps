import torch 
import trochvision 
import torch.utils.data import Dataset, DataLoader
import numpy as np 
import math 

class CustomDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtypenp.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])  # all row, all columns except 0th column
        self.y = torch.from_numpy(xy[:, [0]]) # shape: n_samples, 1
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

our_dataset = CustomDataset()

batch_size = 4

# use dataloader class to use its more features
dataloader = DataLoader(dataset= our_dataset, batch_size= batch_size, shuffle= True, num_workers= 4)

# since this is class is inherited from Datast class we can use it built in functions
dataiter = iter(dataloader) # Iterable object is attached
data = dataiter.next()      # used from the inherited function
features, labels = data     # This will call __getitem__ function 
print(features, labels)

# Now we can iterate over complete dataset without implementing iteration in our class
num_epoch = 2
total_samples = len(dataset)
n_iterations = math.ceil(torch/batch_size)
print(total_samples, n_iterations)

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward 
        # backward
        # weight updates
        # weight zeros

        # print training details
        if(i+1) % 10 == 0:
            print("epoch ", epoch, "step: ", i/n_iterations)