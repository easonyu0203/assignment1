import numpy as np

from algorithms.nmf_base import NMFBase


class SparseNMF(NMFBase):
    """
    Non-negative Matrix Factorization (NMF) with sparsity constraints.

    Implements Sparse NMF by introducing L1 regularization on factor matrices to induce sparsity.

    Attributes:
    - alpha: Regularization parameter for the H matrix.
    - beta: Regularization parameter for the W matrix.
    """

    def __init__(self, V: np.ndarray, num_features: int, alpha: float = 1.0, beta: float = 1.0, **kwargs):
        super().__init__(V, num_features, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def update_step(self) -> None:
        """
        Implements the multiplicative update rules for Sparse NMF with L1 regularization.
        """
        # Update for H
        numerator_H = np.dot(self.W.T, self.V)
        denominator_H = np.dot(self.W.T, np.dot(self.W, self.H)) + self.alpha + self.epsilon
        self.H *= numerator_H / denominator_H

        # Update for W
        numerator_W = np.dot(self.V, self.H.T)
        denominator_W = np.dot(self.W, np.dot(self.H, self.H.T)) + self.beta + self.epsilon
        self.W *= numerator_W / denominator_W

    def metrics(self) -> dict:
        """
        Extends the base metrics function to include L1 regularization in the metrics.
        """
        base_metrics = super().metrics()
        w_reg = np.linalg.norm(self.W, ord=1)
        h_reg = np.linalg.norm(self.H, ord=1)
        base_metrics["W Regularization"] = w_reg
        base_metrics["H Regularization"] = h_reg
        base_metrics["Cost Function"] = np.sum(np.sqrt((self.V - self.W @ self.H) ** 2)) / 2 \
                                        + self.alpha * h_reg + self.beta * w_reg
        return base_metrics
