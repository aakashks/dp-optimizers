from abc import abstractmethod

import numpy as np
import torch


class Accountant:
    def __init__(self, sigma, q, delta=1e-5, max_moment=32):
        self.sigma = sigma
        self.q = q
        self.T = 0
        self.delta = delta
        self.max_moment = max_moment

        # ensure that these conditions are met
        assert sigma >= 1
        assert q < 1 / sigma

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


class MomentsAccountant(Accountant):
    """
    Computes epsilon by the moments accountant method
    as described by

    Abadi, Martín, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang.
    “Deep Learning with Differential Privacy.” In Proceedings of the 2016 ACM SIGSAC
    Conference on Computer and Communications Security, 308–18, 2016.
    https://doi.org/10.1145/2976749.2978318.
    """
    def __init__(self, sigma, q, delta=1e-5, max_moment=32):
        super().__init__(sigma, q, delta, max_moment)

    def _compute_epsilon(self, T):
        # order of moments
        lambda_ = min(self.max_moment, 1 / (10 * self.q ** 2))  # lambda must be <= 32

        epsilon = 3 * self.q * np.sqrt(self.T * np.log(1/self.delta)) / self.sigma

        # epsilon must be greater than this value
        return max(0.002 * T / lambda_, epsilon)


class ModifiedMomentsAccountant(Accountant):
    """
    Computes epsilon using the modified moments accountant method.

    as described in the paper:
    Ding, Xiaofeng, Lin Chen, Pan Zhou, Wenbin Jiang, and Hai Jin.
    “Differentially Private Deep Learning with Iterative Gradient Descent Optimization.”
    ACM/IMS Transactions on Data Science 2, no. 4 (November 30, 2021): 1–27. https://doi.org/10.1145/3491254.
    """
    def __init__(self, sigma, q, delta=1e-5, max_moment=32):
        super().__init__(sigma, q, delta, max_moment)

    def _compute_epsilon(self, T):
        # order of moments
        lambda_ = min(self.max_moment, 1 / (10 * self.q ** 2))  # lambda must be <= 32

        epsilon = 2 * self.q * np.log(1/self.delta) / (self.sigma * np.sqrt(np.power(self.delta, -1/T) - 1))

        # epsilon must be greater than this value
        return max(0.002 * T / lambda_, epsilon)
