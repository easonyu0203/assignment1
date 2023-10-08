import torch

from algorithms.gpu.nmf_base import NMFBase


class MultiplicativeUpdateNMF(NMFBase):
    """
    NMF using multiplicative update rules.
    """

    def update_step(self):
        numerator = torch.mm(self.W.t(), self.V)
        denominator = torch.mm(self.W.t(), torch.mm(self.W, self.H)) + self.epsilon
        self.H *= numerator / denominator

        numerator = torch.mm(self.V, self.H.t())
        denominator = torch.mm(self.W, torch.mm(self.H, self.H.t())) + self.epsilon
        self.W *= numerator / denominator

    def metrics(self) -> dict:
        """
        Extends the base metrics function to include Cost Function
        """
        base_metrics = super().metrics()
        base_metrics["Cost Function"] = (torch.sum(torch.sqrt((self.V - self.W @ self.H) ** 2)) / 2).item()
        return base_metrics
