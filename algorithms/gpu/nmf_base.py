import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score

from utils.early_stopping import EarlyStopping


class NMFBase:
    def __init__(self, V: torch.Tensor, num_features: int, max_iters: int = 1000, epsilon: float = 1e-10):
        self.V = V.cuda()
        self.num_features = num_features
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.W = torch.abs(torch.randn(V.shape[0], num_features, device='cuda'))
        self.H = torch.abs(torch.randn(num_features, V.shape[1], device='cuda'))

    def update_step(self) -> None:
        raise NotImplementedError

    def metrics(self) -> dict:
        return {
            "Reconstruction RMSE": torch.sqrt(torch.mean((self.V - self.W @ self.H) ** 2)).item(),
            "Cost Function": -1
        }

    def fit(self, plot_metrics: bool = False, use_tqdm: bool = True, early_stop: bool = True, patience: int = 10,
            tol: float = 1e-4) -> None:
        metrics_values = {metric: [] for metric in self.metrics().keys()}
        metrics_keys = self.metrics().keys()
        early_stopping = EarlyStopping(patience, tol) if early_stop else None

        iterator = tqdm(range(self.max_iters), desc=f"{self.__class__.__name__} Progress") if use_tqdm else range(self.max_iters)
        for _ in iterator:
            self.update_step()
            current_metrics = self.metrics()
            for metric in metrics_keys:
                metrics_values[metric].append(current_metrics.get(metric, 0))
            if use_tqdm:
                iterator.set_postfix(current_metrics)
            if early_stop and early_stopping.stop(current_metrics["Reconstruction RMSE"]):
                if use_tqdm:
                    iterator.set_postfix({"Status": "Early Stopping", "Best Error": f"{early_stopping.best_error:.4f}"})
                break

        if plot_metrics:
            self.plot_metrics(metrics_values)

    def plot_metrics(self, metrics_values: dict) -> None:
        num_metrics = len(metrics_values)
        rows = (num_metrics + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, (metric_name, values) in enumerate(metrics_values.items()):
            r, c = divmod(idx, 2)
            axes[r, c].plot(values)
            axes[r, c].set_title(f'{metric_name} vs. Iteration Step in {self.__class__.__name__}')
            axes[r, c].set_xlabel('Iteration Step')
            axes[r, c].set_ylabel('Metric Value')

        if num_metrics % 2 != 0:
            fig.delaxes(axes[rows - 1, 1])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _assign_cluster_label(X, Y):
        kmeans = KMeans(n_clusters=len(set(Y)), n_init='auto').fit(X)
        Y_pred = torch.zeros(Y.shape, device='cuda')
        for i in set(kmeans.labels_):
            ind = kmeans.labels_ == i
            Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]  # assign label.
        return Y_pred.cpu().numpy()

    def evaluate(self, Y: torch.Tensor) -> dict:
        R = self.W @ self.H
        Y_pred = NMFBase._assign_cluster_label(R.T.cpu().numpy(), Y.cpu().numpy())
        acc = accuracy_score(Y.cpu().numpy(), Y_pred)
        nmi = normalized_mutual_info_score(Y.cpu().numpy(), Y_pred)
        rmse = torch.sqrt(torch.mean((R - self.V) ** 2)).item()
        return {'acc': acc, "nmi": nmi, "rmse": rmse}

    def get_reconstruction(self) -> torch.Tensor:
        return self.W @ self.H
