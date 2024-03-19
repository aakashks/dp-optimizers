from abc import abstractmethod
from collections import defaultdict
from typing import Literal, List, Dict
import torch
from torch.optim import Optimizer


class DPOptimizer(Optimizer):
    """
    Base class for all differential privacy based optimizers
    Note that this is an abstract class

    Implements Gradient Clipping (with per sample gradients)
    """

    def __init__(self, named_params, lr, max_grad_norm) -> None:
        self.named_params: Dict[str, torch.Tensor] = dict(named_params)
        super().__init__(self.named_params.values(), {})
        self.lr = lr
        self.max_grad_norm = max_grad_norm  # C

    @torch.no_grad()
    def _average_and_clip_grads(self, per_sample_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Averages the per sample gradients and clips them to preserve differential privacy
        :param per_sample_grads: gradients for each sample of the lot (batch)
        :return: clipped and averaged gradients
        """
        new_grads = {
            name: (
                    grad / max(1, torch.norm(grad) / self.max_grad_norm)
            ).mean(0)
            for name, grad in per_sample_grads.items()
        }

        return new_grads


class DPSGD(DPOptimizer):
    """
    Differentiable Private Stochastic Gradient Descent

    as described in the paper:
    Abadi, Martín, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang.
    “Deep Learning with Differential Privacy.”
    In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, 308–18, 2016.
    https://doi.org/10.1145/2976749.2978318.
    """

    def __init__(self, named_params, lot_size, lr=1e-3, noise_scale=4, max_grad_norm=4, weight_decay=0.) -> None:
        super().__init__(named_params, lr, max_grad_norm)
        self.noise_std = noise_scale * max_grad_norm / lot_size  # standard deviation of the gaussian noise added to the gradients
        self.wd = weight_decay

    def _add_noise(self, grad: torch.Tensor) -> torch.Tensor:
        """
        adds gaussian noise to the gradients
        """
        return grad + torch.normal(0, self.noise_std, grad.shape, device=grad.device)

    @torch.no_grad()
    def step(self, per_sample_grads: Dict[str, torch.Tensor]):
        clipped_grads = self._average_and_clip_grads(per_sample_grads)
        for name, param in self.named_params.items():
            if param.requires_grad:
                g = self._add_noise(clipped_grads[name])
                param.copy_((1 - self.wd) * param.detach() - self.lr * g)


class PIGDO(DPOptimizer):
    """
    Perturbed Iterative Gradient Descent Optimizer
    Abstract class for Differential Privacy GDO Algorithms (Adagrad, RMSprop, Adam)

    as described in the paper:
    Ding, Xiaofeng, Lin Chen, Pan Zhou, Wenbin Jiang, and Hai Jin.
    “Differentially Private Deep Learning with Iterative Gradient Descent Optimization.”
    ACM/IMS Transactions on Data Science 2, no. 4 (November 30, 2021): 1–27. https://doi.org/10.1145/3491254.
    """

    def __init__(self, named_params, lot_size, lr, betas, noise_scale, max_grad_norm, weight_decay, eps) -> None:
        super().__init__(named_params, lr, max_grad_norm)
        self.noise_std = noise_scale * max_grad_norm / lot_size  # standard deviation of the gaussian noise added to the gradients
        self.beta1, self.beta2 = betas
        self.wd = weight_decay
        self.eps = eps
        self.state = defaultdict(dict)
        self.t = 0  # time step (used in Adam)

        # state dict to keep track of the moving averages / previous gradients
        for name, param in self.named_params.items():
            self.state[name]['exp_avg'] = torch.zeros_like(param)  # f as in the paper
            self.state[name]['exp_avg_sq'] = torch.zeros_like(param)  # s as in the paper

    def _add_noise(self, grad: torch.Tensor) -> torch.Tensor:
        """
        adds gaussian noise to the gradients
        """
        return grad + torch.normal(0, self.noise_std, grad.shape, device=grad.device)

    @abstractmethod
    def _gdo_rule(self, state: Dict[str, torch.Tensor], grad: torch.Tensor) -> torch.Tensor:
        """
        Gradient Descent Optimization Rule for each algorithm
        will be implemented in the subclasses
        """
        raise NotImplementedError

    @torch.no_grad()
    def step(self, per_sample_grads: Dict[str, torch.Tensor]) -> None:
        # do gradient clipping to preserve differential privacy and then average over the lot
        clipped_grads = self._average_and_clip_grads(per_sample_grads)

        for name, grad in clipped_grads.items():
            # get the state for the parameter
            state = self.state[name]
            grad_hat = self._gdo_rule(state, grad)

            # update gradient according to the algorithm's generated grad_hat
            grad.copy_(grad_hat)

        self.t += 1

        for name, param in self.named_params.items():
            if param.requires_grad:
                # add noise to the gradients
                g = self._add_noise(clipped_grads[name])

                # update the parameters
                param.copy_((1 - self.wd) * param.detach() - self.lr * g)


class PIAdagrad(PIGDO):
    """
    Perturbed Iterative Adagrad Optimizer
    """
    def __init__(self, named_params, lot_size, lr=1e-2, noise_scale=4, max_grad_norm=4,
                 weight_decay=0., eps=1e-10):
        super().__init__(named_params, lot_size, lr, (None, None), noise_scale, max_grad_norm, weight_decay, eps)

    def _gdo_rule(self, state: Dict[str, torch.Tensor], grad: torch.Tensor) -> torch.Tensor:
        state['exp_avg_sq'].add_(grad.square())
        grad_hat = grad / (state['exp_avg_sq'].sqrt() + self.eps)
        return grad_hat


class PIRMSprop(PIGDO):
    """
    Perturbed Iterative RMSprop Optimizer
    """
    def __init__(self, named_params, lot_size, lr=1e-2, beta=0.99, noise_scale=4, max_grad_norm=4,
                 weight_decay=0., eps=1e-8):
        super().__init__(named_params, lot_size, lr, (None, beta), noise_scale, max_grad_norm, weight_decay, eps)

    def _gdo_rule(self, state: Dict[str, torch.Tensor], grad: torch.Tensor) -> torch.Tensor:
        state['exp_avg_sq'].mul_(self.beta2).add_(grad.square(), alpha=1 - self.beta2)

        exp_avg_sq_hat = state['exp_avg_sq']

        grad_hat = grad / (exp_avg_sq_hat.sqrt() + self.eps)
        return grad_hat


class PIAdam(PIGDO):
    """
    Perturbed Iterative Adam Optimizer
    """
    def __init__(self, named_params, lot_size, lr=1e-3, betas=(0.9, 0.999), noise_scale=4, max_grad_norm=4,
                 weight_decay=0., eps=1e-8):
        super().__init__(named_params, lot_size, lr, betas, noise_scale, max_grad_norm, weight_decay, eps)

    def _gdo_rule(self, state: Dict[str, torch.Tensor], grad: torch.Tensor) -> torch.Tensor:
        state['exp_avg'].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        state['exp_avg_sq'].mul_(self.beta2).add_(grad.square(), alpha=1 - self.beta2)

        bias_correction1 = 1 - self.beta1 ** (self.t + 1)
        exp_avg_hat = state['exp_avg'] / bias_correction1

        bias_correction2 = 1 - self.beta2 ** (self.t + 1)
        exp_avg_sq_hat = state['exp_avg_sq'] / bias_correction2

        grad_hat = exp_avg_hat / (exp_avg_sq_hat.sqrt() + self.eps)
        return grad_hat
