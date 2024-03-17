# import MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.func import functional_call, vmap, grad

from dp import optim

class LinearNet(nn.Module):
    def __init__(self, in_features=784):
        super().__init__()
        self.fc = nn.Linear(in_features, 10)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def train_dp_model(model, loss_func, optimizer, num_epochs, num_batches, train_loader, logger):
    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch,))
        loss = loss_func(predictions, targets)
        return loss

    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))


    for epoch in range(num_epochs):
        t_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            output = model(images)
            loss = loss_func(output, labels)

            # backpropagation and optimization
            optimizer.zero_grad()
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, images, labels)
            optimizer.step(ft_per_sample_grads)

            t_loss += loss.detach().cpu().numpy()

            # logging
            if (i+1) % 1 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{num_batches}, Loss {loss:.4f}')

            logger['loss'].append(loss.detach().cpu().numpy())

        logger['total_loss'].append(t_loss / num_batches)



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # data loaders

    train_dataset = datasets.MNIST(
        root='data', download=True, train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(
        root='data', download=True, train=False, transform=transforms.ToTensor())

    q = None # 0.01
    lot_size = int(q * len(train_dataset)) if q is not None else 4000

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=lot_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=lot_size, shuffle=False)

    model = LinearNet().to(device)

    lr = 1e-3

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.DPSGD(model.named_parameters(), lot_size, lr=lr, noise_scale=2, max_grad_norm=4)

    num_epochs = 10
    num_batches = len(train_loader)

    logger = {'loss': [], 'total_loss': []}

    train_dp_model(model, criterion, optimizer, num_epochs, num_batches, train_loader, logger)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(logger['loss'])
    ax[0].set_title('loss')

    ax[1].plot(logger['total_loss'])
    ax[1].set_title('total loss')
    plt.show()
