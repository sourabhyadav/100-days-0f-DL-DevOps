import torch
import torch.nn as nn 

FILE = "model.pth"

class tmpModel(nn.Module):
    def __init__(self, n_input):
        super(tmpModel, self).__init__()
        self.linear = nn.Linear(n_input, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = tmpModel(n_input=6)

# Training your model then save it

# SAVING the MODEL

# METHOD #1: Lazy method. This is not recommended
#torch.save(model, FILE)

# load the model
# model = torch.load(FILE)
# model.eval()

# for param in model.parameters():
#     print(param)

# METHOD #2: Save dict method. recommended
# torch.save(model.state_dict(), FILE)

# load the model
# loaded_model = tmpModel(n_input=6)
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()

# for param in model.parameters():
#     print(param)

# We can also save further dicts 
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
print(optimizer.state_dict())

# this could be useful to strop training and resume witht the something checkpoint
checkpoint = {
    "epoch" : 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

# torch.save(checkpoint, "checkpoint.pth")

# Loading the checkpoint dict as a whole
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = tmpModel(n_input=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

print("After loading optimizer: ", optimizer.state_dict())