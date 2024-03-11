# import MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from dp import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# data loaders

train_dataset = datasets.MNIST(
    root='data', download=True, train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(
    root='data', download=True, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=512, shuffle=False)

class LinearNet(nn.Module):
    def __init__(self, in_features=784):
        super().__init__()
        self.fc = nn.Linear(in_features, 10)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def train_model(model, num_epochs, num_batches, train_loader, logger):
    for epoch in range(num_epochs):
        t_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            output = model(images)
            loss = criterion(output, labels)

            # backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.detach().cpu().numpy()

            # logging
            if (i+1) % 1 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{num_batches}, Loss {loss:.4f}')

            logger['loss'].append(loss.detach().cpu().numpy())

        logger['total_loss'].append(t_loss / num_batches)



if __name__ == '__main__':

    model = LinearNet().to(device)

    lr = 1e-3

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    num_epochs = 4
    num_batches = len(train_loader)

    logger = {'loss': [], 'total_loss': []}

    train_model(model, num_epochs, num_batches, train_loader, logger)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(logger['loss'])
    ax[0].set_title('loss')

    ax[1].plot(logger['total_loss'])
    ax[1].set_title('total loss')
    plt.show()
