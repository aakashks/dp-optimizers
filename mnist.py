import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from torchvision import datasets, transforms

import click
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dp import optim, train_dp_model


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
@click.option('--num-epochs', default=50, help='Number of epochs.')
@click.option('--lot-size', default=2000, help='Lot size.')
@click.option('--lr', default=0.05, help='Learning rate.')
@click.option('--noise-scale', default=2, help='Noise scale.')
@click.option('--max-grad-norm', default=4, help='Max gradient norm.')
@click.option('--q', default=None, help='q ratio (use if not using lot-size).')
@click.option('--hidden-size', default=1000, help='Hidden size.')
@click.option('--save-fig', is_flag=True, default=False, help='Save figure.')
@click.option('--device', default=None, help='Device.')
def run_mnist(num_epochs, lot_size, lr, noise_scale, max_grad_norm, q, hidden_size, save_fig, device):
    # set device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    ) if device is None else device

    # data loaders
    train_data = datasets.MNIST(
        root='data', download=True, train=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(
        root='data', download=True, train=False, transform=transforms.ToTensor())

    # apply PCA to the dataset (as done in the paper)
    X_train = train_data.train_data.numpy().reshape(len(train_data), -1)
    X_test = test_data.test_data.numpy().reshape(len(test_data), -1)

    pca = PCA(n_components=60)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # form tensors
    X_train_pca_tensor = torch.from_numpy(X_train_pca).float()
    X_test_pca_tensor = torch.from_numpy(X_test_pca).float()
    y_train = torch.tensor(train_data.train_labels.numpy())
    y_test = torch.tensor(test_data.test_labels.numpy())

    # create torch datasets
    train_dataset = TensorDataset(X_train_pca_tensor, y_train)
    test_dataset = TensorDataset(X_test_pca_tensor, y_test)

    # training settings
    lot_size = lot_size if q is None else int(q * len(train_dataset))  # (L)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=lot_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=lot_size, shuffle=False)

    model = LinearNet(in_features=pca.n_components, hidden=hidden_size)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # differentially private optimizer
    optimizer = optim.DPSGD(model.named_parameters(), lot_size, lr=lr, noise_scale=noise_scale,
                            max_grad_norm=max_grad_norm)

    num_batches = len(train_loader)

    logger = {'loss': [], 'total_loss': [], 'accuracy': [], 'total_accuracy': [], 'total_val_accuracy': []}

    train_dp_model(model, criterion, optimizer, num_epochs, num_batches, train_loader, test_loader, device=device,
                   logger=logger)

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
