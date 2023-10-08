import torch
import numpy as np

from algorithms.gpu.nmf_base import NMFBase


class SparseNMF(NMFBase):
    """
    Non-negative Matrix Factorization (NMF) with sparsity constraints.

    Implements Sparse NMF by introducing L1 regularization on factor matrices to induce sparsity.

    Attributes:
    - alpha: Regularization parameter for the H matrix.
    - beta: Regularization parameter for the W matrix.
    """

    def __init__(self, V: np.array, num_features: int, alpha: float = 1.0, beta: float = 1.0, **kwargs):
        super().__init__(V, num_features, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def update_step(self, current_iter: int) -> None:
        """
        Implements the multiplicative update rules for Sparse NMF with L1 regularization.
        """
        # Update for H
        numerator_H = torch.mm(self.W.t(), self.V)
        denominator_H = torch.mm(self.W.t(), torch.mm(self.W, self.H)) + self.alpha + self.epsilon
        self.H *= numerator_H / denominator_H

        # Update for W
        numerator_W = torch.mm(self.V, self.H.t())
        denominator_W = torch.mm(self.W, torch.mm(self.H, self.H.t())) + self.beta + self.epsilon
        self.W *= numerator_W / denominator_W

    def metrics(self) -> dict:
        """
        Extends the base metrics function to include L1 regularization in the metrics.
        """
        base_metrics = super().metrics()
        w_reg = torch.norm(self.W, p=1)
        h_reg = torch.norm(self.H, p=1)
        base_metrics["W Regularization"] = w_reg.item()
        base_metrics["H Regularization"] = h_reg.item()
        base_metrics["Cost Function"] = (torch.sum(torch.sqrt((self.V - self.W @ self.H) ** 2)) / 2
                                        + self.alpha * h_reg + self.beta * w_reg).item()
        return base_metrics
