from abc import abstractmethod
from typing import Literal, List, Dict
import torch
from torch.optim import Optimizer


class DPOptimizer(Optimizer):
    """
    Base class for all optimizers
    Note that this is an abstract class

    Does Gradient Clipping (with per sample gradients)
    """

    def __init__(self, named_params, lr, max_grad_norm) -> None:
        self.named_params: Dict[str, torch.Tensor] = dict(named_params)
        super().__init__(self.named_params.values(), {})
        self.lr = lr
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def _average_and_clip_grads(self, per_sample_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    """

    def __init__(self, named_params, lot_size, lr=1e-3, noise_scale=4, max_grad_norm=4, weight_decay=0.) -> None:
        super().__init__(named_params, lr, max_grad_norm)
        self.noise_std = noise_scale * max_grad_norm / lot_size
        self.wd = weight_decay

    def _add_noise(self, grad: torch.Tensor) -> torch.Tensor:
        return grad + torch.normal(0, self.noise_std, grad.shape, device=grad.device)

    @torch.no_grad()
    def step(self, per_sample_grads: Dict[str, torch.Tensor]):
        clipped_grads = self._average_and_clip_grads(per_sample_grads)
        for name, param in self.named_params.items():
            if param.requires_grad:
                g = self._add_noise(clipped_grads[name])
                param.copy_((1 - self.wd) * param.detach() - self.lr * g)
