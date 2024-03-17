import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from torchvision import datasets, transforms

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


if __name__ == '__main__':

    # set device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

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
    num_epochs = 20  # 1/q
    # q = None  # 0.01
    lot_size = 600  # if q is None else int(q * len(train_dataset))  # (L)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=lot_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=lot_size, shuffle=False)

    model = LinearNet(in_features=pca.n_components).to(device)

    lr = 0.05

    # loss function
    criterion = nn.CrossEntropyLoss()

    # differentially private optimizer
    optimizer = optim.DPSGD(model.named_parameters(), lot_size, lr=lr, noise_scale=2, max_grad_norm=4)

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=2, total_iters=10)

    num_batches = len(train_loader)

    logger = {'loss': [], 'total_loss': [], 'accuracy': [], 'total_accuracy': []}

    train_dp_model(model, criterion, optimizer, num_epochs, num_batches, train_loader, scheduler, device, logger)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(logger['accuracy'])
    ax[0].set_title('accuracy')

    ax[1].plot(logger['total_accuracy'])
    ax[1].set_title('total accuracy')
    plt.show()
