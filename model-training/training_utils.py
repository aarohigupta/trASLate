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
        output: 24
        """
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)       # 1 input channel, 6 output channels, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)        # 2x2 kernel
        self.conv2 = nn.Conv2d(6, 6, 3)       # 6 input channels, 6 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(6, 16, 3)      # 6 input channels, 16 output channels, 3x3 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 400 input channels, 120 output channels
        self.fc2 = nn.Linear(120, 48)         # 120 input channels, 48 output channels
        self.fc3 = nn.Linear(48, 24)          # 48 input channels, 24 output channels
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# define a function to train the model
def train_model(model, train_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(num_epochs):
        curr_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = Variable(data['image'].float()), Variable(data['label'].long())
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels[:, 0]) # labels[:, 0] because labels is a 2D tensor
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()
            if i % 100 == 0:
                print(f'epoch: {epoch}, batch: {i}, loss: {curr_loss / (i + 1)}')
        curr_loss = 0.0
        schedular.step()

def train(model_name: str, num_epochs: int, train_loader: Dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet().float().to(device)
    train_model(model, train_loader, num_epochs, device)
    torch.save(model.state_dict(), f'models/{model_name}.pth')