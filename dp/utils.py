from torch.func import functional_call, vmap, grad
import torch


def train_dp_model(model, loss_fn, optimizer, num_epochs, num_batches, train_loader, val_loader, scheduler=None, device=torch.device('cpu'), logger=None):
    model.to(device)

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch,))
        loss = loss_fn(predictions, targets)
        return loss

    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    for epoch in range(num_epochs):
        total_loss = 0
        total_acc = 0
        total_val_acc = 0
        for i, ((images, labels), (val_images, val_labels)) in enumerate(zip(train_loader, val_loader)):
            # move images and labels to device
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            # backpropagation and optimization
            optimizer.zero_grad()
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, images, labels)
            optimizer.step(ft_per_sample_grads)

            with torch.no_grad():
                model.eval()
                # forward pass
                output = model(images)
                # storing loss
                # loss = loss_fn(output, labels)
                # total_loss += loss.detach().cpu().numpy()

                # storing accuracy
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == labels).sum().item() / labels.size(0)
                total_acc += accuracy

                # validation accuracy
                val_output = model(val_images.to(device))
                _, val_predicted = torch.max(val_output.data, 1)
                val_accuracy = (val_predicted == val_labels.to(device)).sum().item() / val_labels.size(0)
                total_val_acc += val_accuracy

            # logging
            if (i + 1) % 1 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{num_batches}, Train Acc: {accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

            # logger['loss'].append(loss.detach().cpu().numpy())
            logger['accuracy'].append(accuracy)

        if scheduler is not None:
            scheduler.step()
        # logger['total_loss'].append(total_loss / num_batches)
        logger['total_accuracy'].append(total_acc / num_batches)
        logger['total_val_accuracy'].append(total_val_acc / num_batches)
