import numpy as np


class EarlyStopping:
    def __init__(self, patience, tol):
        self.patience = patience
        self.tol = tol
        self.best_error = np.inf
        self.patience_counter = 0

    def stop(self, current_error):
        if current_error + self.tol < self.best_error:
            self.best_error = current_error
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience
