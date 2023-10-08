import numpy as np

from algorithms.nmf_base import NMFBase
from utils.lr_scheduler import LearningRateScheduler


class RobustNMF(NMFBase):
    """
    Non-negative Matrix Factorization (NMF) with robustness against outliers and noise.

    Implements Robust NMF which models noise/errors explicitly in a matrix E.

    Attributes:
    - lambda_param: Regularization parameter for the soft-thresholding operation on the error matrix.
    - learning_rate: Learning rate for gradient descent updates.
    - E: Error matrix representing noise or outliers.
    """

    def __init__(self, V: np.array, num_features: int, lambda_param: float, start_learning_rate: float = 0.001,
                 end_learning_rate: float = 0.0001, **kwargs):
        super().__init__(V, num_features, **kwargs)
        self.lambda_param = lambda_param
        self.scheduler = LearningRateScheduler(start_learning_rate, end_learning_rate, self.max_iters)
        self.learning_rate = start_learning_rate
        self.E = np.zeros_like(V)

    def update_step(self, current_iter: int) -> None:
        """
        Implements the gradient descent update rules for Robust NMF.
        """
        WH = np.dot(self.W, self.H)
        gradient_W = -np.dot(self.V - WH - self.E, self.H.T)
        gradient_H = -np.dot(self.W.T, self.V - WH - self.E)

        # Use learning rate from the scheduler
        self.learning_rate = self.scheduler.get_lr(current_iter)
        self.W = self.W - self.learning_rate * gradient_W
        self.H = self.H - self.learning_rate * gradient_H

        # Soft thresholding for E
        residual = self.V - np.dot(self.W, self.H)
        self.E = np.sign(residual) * np.maximum(np.abs(residual) - self.lambda_param, 0)

    def metrics(self) -> dict:
        """
        Extends the base metrics function to include the norm of the error matrix.
        """
        base_metrics = super().metrics()
        base_metrics["||Error Matrix||"] = np.sum(np.sqrt(self.E ** 2))
        base_metrics["Cost Function"] = np.sum(np.sqrt((self.V - self.W @ self.H - self.E) ** 2)) / 2 + \
                                        self.lambda_param * np.sum(np.abs(self.E))
        return base_metrics
