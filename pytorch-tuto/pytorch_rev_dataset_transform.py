import torch 
import trochvision 
import torch.utils.data import Dataset, DataLoader
import numpy as np 
import math 

class CustomDataset(Dataset):
    # now out dataset should support transform class of Dataset
    def __init__(self, transform= None):    # we can pass custom transform function
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtypenp.float32, skiprows=1)
        self.x = xy[:, 1:]  # all row, all columns except 0th column
        self.y = xy[:, [0]] # shape: n_samples, 1
        self.n_samples = xy.shape[0]

        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample) # basically it is a custom function that we can pass

        return sample

    def __len__(self):
        return self.n_samples

# implement our own transform functions
class ToTensor():
    def __call__(self, sample): # this method is used to make a call when it is passed to other function
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)


our_dataset = CustomDataset(transform=ToTensor())
first_data = our_dataset[0]     # __getitem__ function is called
features, labels = first_data   # this will be a tranfromed dataset


# We can also combine multiple custom transfrom 

class MultiTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(sefl, sample):
        inputs, traget = sample
        inputs += self.factor
        return inputs, target 

# combine multiple transforms here
composed = torchvision.tranforms.Compose([ToTensor(), MultiTransform(4))    # provide multiple transforms as a list
dataset = CustomDataset(transform=composed)
first_data = our_dataset[0]
features, labels = first_data # this will be a tranfromed by both the provided transforms
