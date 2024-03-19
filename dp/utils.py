from torch.func import functional_call, vmap, grad
import torch


def train_dp_model(model, loss_fn, optimizer, num_epochs, train_loader, val_loader, device=torch.device('cpu'), logger=None):
    # get length of train and val loaders
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)

    # compute per-sample-gradients efficiently by using function transforms

    # extract the state from model
    # detach as we won't use torch.autograd / Tensor.backward()
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    # function to compute the loss for each sample
    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        # treat nn.Module as a function
        predictions = functional_call(model, (params, buffers), (batch,))
        loss = loss_fn(predictions, targets)
        return loss

    # grad transform to create a function that computes gradient with respect to the first argument of compute_loss
    ft_compute_grad = grad(compute_loss)
    # use vmap to vectorize the function over the whole batch of samples
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    # training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_acc = 0
        total_val_acc = 0

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                # forward pass
                output = model(images)
                # storing loss
                # loss = loss_fn(output, labels)
                # total_loss += loss.detach().cpu().numpy()

                # storing accuracy
                _, predicted = torch.max(output, 1)
                accuracy = (predicted == labels).sum().item() / labels.size(0)
                total_acc += accuracy

            # optimization step
            optimizer.zero_grad()
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, images, labels)
            optimizer.step(ft_per_sample_grads)

            # logging
            if (i + 1) % 1 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{len_train_loader}, Train Acc: {accuracy:.4f}')

            # logger['loss'].append(loss.detach().cpu().numpy())
            logger['accuracy'].append(accuracy)

        # validation check
        with torch.no_grad():
            model.eval()
            for i, (val_images, val_labels) in enumerate(val_loader):

                # validation accuracy
                val_output = model(val_images.to(device))
                _, val_predicted = torch.max(val_output, 1)
                val_accuracy = (val_predicted == val_labels.to(device)).sum().item() / val_labels.size(0)
                total_val_acc += val_accuracy

        # logger['total_loss'].append(total_loss / num_batches)
        logger['total_accuracy'].append(total_acc / len_train_loader)
        logger['total_val_accuracy'].append(total_val_acc / len_val_loader)
