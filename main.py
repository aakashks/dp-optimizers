import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from dp import optim, train_dp_model


class LinearNet(nn.Module):
    def __init__(self, in_features=784):
        super().__init__()
        self.hidden = nn.Linear(in_features, 1000)
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.hidden(x))
        x = self.fc(x)
        return x


if __name__ == '__main__':

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    # data loaders
    train_dataset = datasets.MNIST(
        root='data', download=True, train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(
        root='data', download=True, train=False, transform=transforms.ToTensor())

    q = None  # 0.01
    lot_size = int(q * len(train_dataset)) if q is not None else 6000

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=lot_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=lot_size, shuffle=False)

    model = LinearNet().to(device)

    lr = 1e-3

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.DPSGD(model.named_parameters(), lot_size, lr=lr, noise_scale=2, max_grad_norm=4)

    num_epochs = 20  # 1/q
    num_batches = len(train_loader)

    logger = {'loss': [], 'total_loss': [], 'accuracy': [], 'total_accuracy': []}

    train_dp_model(model, criterion, optimizer, num_epochs, num_batches, train_loader, device, logger)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(logger['accuracy'])
    ax[0].set_title('accuracy')

    ax[1].plot(logger['total_accuracy'])
    ax[1].set_title('total accuracy')
    plt.show()
