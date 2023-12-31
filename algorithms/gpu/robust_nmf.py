import torch
import numpy as np

from algorithms.gpu.nmf_base import NMFBase
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
        self.E = torch.zeros_like(self.V)

    def update_step(self, current_iter: int) -> None:
        """
        Implements the gradient descent update rules for Robust NMF.
        """

        WH = torch.mm(self.W, self.H)
        gradient_W = -torch.mm(self.V - WH - self.E, self.H.t())
        gradient_H = -torch.mm(self.W.t(), self.V - WH - self.E)

        # Use learning rate from the scheduler
        self.learning_rate = self.scheduler.get_lr(current_iter)
        self.W = self.W - self.learning_rate * gradient_W
        self.H = self.H - self.learning_rate * gradient_H

        # Soft thresholding for E
        residual = self.V - torch.mm(self.W, self.H)
        self.E = torch.sign(residual) * torch.clamp(torch.abs(residual) - self.lambda_param, min=0)

    def metrics(self) -> dict:
        """
        Extends the base metrics function to include the norm of the error matrix.
        """
        base_metrics = super().metrics()
        base_metrics["||Error Matrix||"] = torch.sum(torch.sqrt(self.E ** 2)).item()
        base_metrics["Cost Function"] = (torch.sum(torch.sqrt((self.V - self.W @ self.H - self.E) ** 2)) / 2 +
                                        self.lambda_param * torch.sum(torch.abs(self.E))).item()
        return base_metrics
