import os

import numpy as np
import torch


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class EarlyStopping:
    def __init__(self, patience=10, verbose=True, min_delta=1e-4, mode="max"):
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _is_better(self, current, best):
        if self.mode == "min":
            return (best - current) > self.min_delta
        return (current - best) > self.min_delta

    def __call__(self, monitor, model, path):
        if self.best_score is None:
            self.best_score = monitor
            self.save_checkpoint(model, path)
            return
        if self._is_better(monitor, self.best_score):
            if self.verbose:
                print(f"Monitor improved ({self.best_score:.6f} -> {monitor:.6f}). Saving checkpoint.")
            self.best_score = monitor
            self.counter = 0
            self.save_checkpoint(model, path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    @staticmethod
    def save_checkpoint(model, path):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
