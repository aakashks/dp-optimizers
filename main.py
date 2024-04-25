import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from torchvision import datasets, transforms

import click
import matplotlib.pyplot as plt

from dpgdo import optim, train_dp_model, accountants


class LinearNet(nn.Module):
    """
    Simple linear model with 1000 hidden units
    """

    def __init__(self, in_features=784, hidden=1000, num_classes=10):
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.hidden(x))
        x = self.fc(x)
        return x


@click.command()
@click.option('--dataset', default='mnist', help='Dataset to use.')
@click.option('--num-epochs', default=10, help='Number of epochs.')
@click.option('--lot-size', default=600, help='Lot size.')
@click.option('--lr', default=0.0001, help='Learning rate.')
@click.option('--noise-scale', default=4, help='Noise scale.')
@click.option('--max-grad-norm', default=4, help='Max gradient norm.')
@click.option('--q', default=None, help='Sampling Probability (use if not using lot-size).')
@click.option('--hidden-size', default=1000, help='Hidden size.')
@click.option('--no-pca', is_flag=True, default=False, help='Do not apply pca to data before applying NN.')
@click.option('--save-fig', is_flag=True, default=False, help='Save figure.')
@click.option('--device', default=None, help='Device.')
def run_mnist(dataset, num_epochs, lot_size, lr, noise_scale, max_grad_norm, q, hidden_size, no_pca, save_fig, device):
    # set device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    ) if device is None else torch.device(device)

    print(f'Using device: {device}')

    # data loaders
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(
            root='data', download=True, train=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(
            root='data', download=True, train=False, transform=transforms.ToTensor())

    else:
        train_dataset = datasets.FashionMNIST(
            root='data', download=True, train=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(
            root='data', download=True, train=False, transform=transforms.ToTensor())

    if not no_pca:
        # apply PCA to the dataset (as done in the paper)
        X_train = train_dataset.data.reshape(len(train_dataset), -1)
        X_test = test_dataset.data.reshape(len(test_dataset), -1)

        A = torch.cat([X_train, X_test]).float()
        pca_dim = 60
        _, _, V = torch.pca_lowrank(A, q=pca_dim)

        res = torch.matmul(A, V)

        X_train_pca_tensor = res[:60000]
        X_test_pca_tensor = res[60000:]
        y_train = train_dataset.targets
        y_test = test_dataset.targets

        # create torch datasets
        train_dataset = TensorDataset(X_train_pca_tensor, y_train)
        test_dataset = TensorDataset(X_test_pca_tensor, y_test)

    # training settings
    lot_size = lot_size if q is None else int(q * len(train_dataset))  # (L)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=lot_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=lot_size, shuffle=False)

    model = LinearNet(in_features=784 if no_pca else pca_dim, hidden=hidden_size).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # differentially private optimizer
    optimizer = optim.PIAdam(model.named_parameters(), lot_size, lr=lr, noise_scale=noise_scale,
                             max_grad_norm=max_grad_norm)

    accountant = accountants.ModifiedMomentsAccountant(noise_scale, q=lot_size / len(train_dataset))

    logger = {'loss': [], 'total_loss': [], 'accuracy': [], 'total_accuracy': [], 'total_val_accuracy': [], 'epsilon': []}

    train_dp_model(model, criterion, optimizer, num_epochs, train_loader, test_loader, device=device,
                   logger=logger, accountant=accountant, verbose=2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(logger['accuracy'])
    ax[0].set_title('accuracy')

    ax[1].plot(logger['total_accuracy'], label='train accuracy')
    ax[1].set_title('total accuracy')
    ax[1].plot(logger['total_val_accuracy'], label='val accuracy')
    ax[1].legend()
    plt.show()

    if save_fig:
        os.makedirs('output', exist_ok=True)
        fig.savefig('output/mnist.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    run_mnist()
