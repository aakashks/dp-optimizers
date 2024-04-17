from torch.func import functional_call, vmap, grad
import torch


def train_dp_model(model, loss_fn, optimizer, num_epochs, train_loader, val_loader, device=torch.device('cpu'),
                   logger=None, accountant=None, verbose=1):
    """
    Train a differentially private model using PyTorch.

    per sample gradients are based upon the reference:
    “Per-Sample-Gradients — PyTorch Tutorials 2.2.1+Cu121 Documentation,”
    Pytorch.org, 2024, https://pytorch.org/tutorials/intermediate/per_sample_grads.html.

    """
    # get length of train and val loaders
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)

    # compute per-sample-gradients efficiently by using function transforms

    # extract model state
    # detach as we won't use torch.autograd / .backward()
    model_params = {k: v.detach() for k, v in model.named_parameters()}
    model_buffers = {k: v.detach() for k, v in model.named_buffers()}

    # function to compute the loss for each sample
    def compute_loss(params, buffers, inputs, labels):
        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)

        # use function call to treat nn.Module as a function
        predictions = functional_call(model, (params, buffers), (inputs,))
        loss = loss_fn(predictions, labels)
        return loss

    # grad transform to make a function that computes gradient of only the first arg of compte_loss
    compute_grad = grad(compute_loss)
    # use vmap to vectorize the function over the whole batch of image samples
    compute_sample_grad = vmap(compute_grad, in_dims=(None, None, 0, 0))

    # training loop
    for epoch in range(num_epochs):
        # total_loss = 0
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

                if epoch == 0 and i == 0:
                    if accountant is not None:
                        logger['epsilon'].append(0)

                    logger['total_accuracy'].append(accuracy)

            # optimization step
            optimizer.zero_grad()
            per_sample_grads = compute_sample_grad(model_params, model_buffers, images, labels)
            optimizer.step(per_sample_grads)

            if accountant is not None:
                accountant.step()  # increment the number of steps

            if verbose > 1:
                # logging
                if (i + 1) % 1 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{len_train_loader}, Train Acc: {accuracy:.4f}')

            # logger['loss'].append(loss.detach().cpu().numpy())
            logger['accuracy'].append(accuracy)

        model.eval()
        # validation check
        with torch.no_grad():
            for i, (val_images, val_labels) in enumerate(val_loader):
                # validation accuracy
                val_output = model(val_images.to(device))
                _, val_predicted = torch.max(val_output, 1)
                val_accuracy = (val_predicted == val_labels.to(device)).sum().item() / val_labels.size(0)
                total_val_acc += val_accuracy

                if epoch == 0 and i == 0:
                    logger['total_val_accuracy'].append(val_accuracy)

        # logger['total_loss'].append(total_loss / len_train_loader)
        logger['total_accuracy'].append(total_acc / len_train_loader)
        logger['total_val_accuracy'].append(total_val_acc / len_val_loader)

        if accountant is not None:
            eps = accountant.get_privacy_spent()
            logger['epsilon'].append(eps)

    if verbose > 0 and accountant is not None:
        print(f"{optimizer.__class__.__name__} run {num_epochs} epochs -> final test accuracy: {logger['total_val_accuracy'][-1]:.4f}, final epsilon: {logger['epsilon'][-1]:.2f} by {accountant.__class__.__name__}")
