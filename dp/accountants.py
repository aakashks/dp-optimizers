from abc import abstractmethod

import torch


class MomentsAccountant:
    def __init__(self, sigma, q, delta):
        self.sigma = sigma
        self.q = q
        self.T = 0
        self.delta = delta

    def step(self):
        self.T += 1

    def get_privacy_spent(self):
        """
        Returns the privacy spent.
        """
        return self._compute_epsilon(self.T)

    @abstractmethod
    def _compute_epsilon(self, T):
        raise NotImplementedError


class ModifiedMomentsAccountant(MomentsAccountant):
    """
    Computes epsilon using the modified moments accountant method.

    as described in the paper:
    Ding, Xiaofeng, Lin Chen, Pan Zhou, Wenbin Jiang, and Hai Jin.
    “Differentially Private Deep Learning with Iterative Gradient Descent Optimization.”
    ACM/IMS Transactions on Data Science 2, no. 4 (November 30, 2021): 1–27. https://doi.org/10.1145/3491254.
    """
    def __init__(self, sigma, q, delta=1e-5):
        super().__init__(sigma, q, delta)

        # ensure conditions from Theorem 3 are met
        assert sigma >= 1
        assert q < 1 / sigma

    def _compute_epsilon(self, T):
        # order of moments
        lambda_ = min(32, 1 / (10 * self.q ** 2))  # lambda must be <= 32

        # the cumulant generating function of the privacy loss at each step of the optimization
        K_g = self.T * torch.log(torch.tensor(1 + (self.q ** 2 * lambda_ ** 2) / self.sigma ** 2))

        epsilon = 0.002 * self.T / lambda_
        assert epsilon > 0, "epsilon must be positive"

        return max(epsilon, 2 * K_g / lambda_)
