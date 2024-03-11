from abc import abstractmethod
from typing import Literal, List
import torch


class Optimizer:
    """
    Base class for all optimizers
    note that this is an abstract class
    """
    def __init__(self, params, lr=1e-3) -> None:
        self.params: List[torch.Tensor] = list(params)
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError()

    @torch.no_grad()
    def zero_grad(self) -> None:
        for param in self.params:
            if param.requires_grad:
                param.grad = None


class SGD(Optimizer):
    """
    base class for SGD based dp optimizers
    change modify_grad to accordingly add noise in the gradients

    Note that here the gradient obtained is aggregated over all minibatch samples
    TODO: Implement per example gradient (for pSGD)
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.) -> None:
        super().__init__(params, lr)
        self.wd = weight_decay

    def modify_grad(self, grad: torch.Tensor) -> torch.Tensor:
        return grad

    @torch.no_grad()
    def step(self):
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                g = self.modify_grad(param.grad.detach())
                param.copy_((1 - self.wd) * param.detach() - self.lr * g)


# class PSGD(SGD):
#     def __init__(self, params, lr=1e-3) -> None:
#         super.__init__(params, lr)
#
#     def modify_grad(self, grad: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError()
#
#
# class Adam(Optimizer):
#     def __init__(self, params, lr, beta1, beta2) -> None:
#         self.beta1 = beta1
#         self.beta2 = beta2
#
#         super.__init__(params, lr)
