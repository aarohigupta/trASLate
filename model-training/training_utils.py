from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from preprocessing_utils import get_train_test_dataloader

# define a class for the neural network that inherits from nn.Module (base class for all neural network modules)
class NeuralNet(nn.Module):
    def __init__(self) -> None:
        """
        Define the layers of the neural network
        input: 1 x 28 x 28
        output: 24 (since we have 24 classes)
        """
        super(NeuralNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        # print the output shape of the convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4, 4) # to reduce the size of the image to 32
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 24)

    def forward(self, x):
        """
        Define the forward pass of the neural network
        """

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# define a function to train the model
# def train_model(model, train_loader, num_epochs, device):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#     schedular = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
#     for epoch in range(num_epochs):
#         curr_loss = 0.0
#         for i, data in enumerate(train_loader, 0):
#             inputs, labels = Variable(data['image'].float()), Variable(data['label'].long())
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels[:, 0]) # labels[:, 0] because labels is a 2D tensor
#             loss.backward()
#             optimizer.step()

#             curr_loss += loss.item()
#             if i % 100 == 0:
#                 print(f'epoch: {epoch}, batch: {i}, loss: {curr_loss / (i + 1)}')
#         curr_loss = 0.0
#         schedular.step()

# def train(model_name: str, num_epochs: int, train_loader: Dataset):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = NeuralNet().float().to(device)
#     train_model(model, train_loader, num_epochs, device)
#     torch.save(model.state_dict(), f'models/{model_name}.pth')

def train(model: nn.Module, train_loader: Dataset, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler, num_epochs: int, device: torch.device):
    """
    @param model: the model to train
    @param train_loader: the dataloader for the training data
    @param criterion: the loss function
    @param optimizer: the optimizer algorithm
    @param scheduler: the learning rate scheduler (for SGD, Adam, etc.)
    @param num_epochs: the number of epochs to train for
    @param device: the device to train on (CPU or GPU)

    @return: the trained model

    Trains the model for the specified number of epochs

    Model is set to train mode, gradients are zeroed, forward pass is performed, loss is calculated and backpropogated, and the optimizer is updated
    """
    for epoch in range(num_epochs):
        model.train()
        curr_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = Variable(data['image'].float()), Variable(data['label'].long())
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            # print(f"images: {inputs.size()}")
            # print(f"labels: {labels.size()}")
            # print(f"outputs: {outputs.size()}")
            loss = criterion(outputs, labels[:, 0])

            # backward pass + optimize
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()
            if i % 100 == 0:
                print(f'epoch: {epoch}, batch: {i}, loss: {curr_loss / (i + 1)}')
        curr_loss = 0.0
        scheduler.step()

    return model