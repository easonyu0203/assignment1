import numpy as np

from algorithms.nmf_base import NMFBase


class MultiplicativeUpdateNMF(NMFBase):
    """
    NMF using multiplicative update rules.
    """

    def update_step(self):
        numerator = np.dot(self.W.T, self.V)
        denominator = np.dot(self.W.T, np.dot(self.W, self.H)) + self.epsilon
        self.H *= numerator / denominator

        numerator = np.dot(self.V, self.H.T)
        denominator = np.dot(self.W, np.dot(self.H, self.H.T)) + self.epsilon
        self.W *= numerator / denominator

    def metrics(self) -> dict:
        """
        Extends the base metrics function to include Cost Function
        """
        base_metrics = super().metrics()
        base_metrics["Cost Function"] = np.sum(np.sqrt((self.V - self.W @ self.H) ** 2)) / 2
        return base_metrics
