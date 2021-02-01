import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as pyplot
import time
import os
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean    = np.array([0.485, 0.456, 0.406])
std     = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ]),
}

# import data
data_dir = 'data/fire_classi'
sets = ['train', 'val']

# parse the dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                    data_transforms[x]) for x in ['train', 'val']}

# wrap around a dataloader
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                shuffle=True, num_workers=0)
                for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes 
print("Class names: ", class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_time = time.time()
        print("Epoch ", epoch, "/", num_epochs)
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0 
            running_corrects = 0

            # iterate over data 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward pass 
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # get statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("phase: ", phase, " loss: ", epoch_loss, " acc: ", epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print("eoch: ", epoch, " Training time: ", time.time() - epoch_time)
        print()
    
    time_elapsed = time.time() - since
    print("Training completed!!! Time taken: ", time_elapsed)
    print("best acc: ", best_acc)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model 

# import pre trained model
model = models.resnet18(pretrained=True)

print("Model layers: ")
for name, param in model.named_parameters():
    print (name)

# freeze other pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the last layer
num_features = model.fc.in_features
print("fc feature size: ", num_features, " , ", model.fc.out_features)
model.fc = nn.Linear(num_features, 2)
print("fc feature size: ", model.fc.in_features, " , ", model.fc.out_features)

model.to(device)

# check if the layers are freezed or not
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler)

