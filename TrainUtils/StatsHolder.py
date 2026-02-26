# Import packages
import numpy as np


# Class
class StatsHolder:

    # Define class attributes
    eps = 1e-7
    table_stats = ["f1", "auc", "mcc", "bal_acc"]
    comparable_stats = {"acc": "Accuracy", "f1": "F1-score", "auc": "AUC", "mcc": "MCC", "bal_acc": "Balanced Accuracy"}

    def __init__(self, loss, acc, tp, tn, fp, fn, auc, extra_stats=None):
        # Initialize attributes
        self.loss = loss
        self.acc = acc
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.auc = auc
        if acc == 1:
            self.eps = 0.0

        # Compute extra stats
        if extra_stats is not None:
            (self.n_vals, self.sens, self.spec, self.precis, self.neg_pred_val, self.f1, self.mcc, self.auc, self.fnr,
             self.fpr) = extra_stats
        else:
            self.n_vals = 1
            self.sens = np.round(tp / (tp + fn + self.eps) + self.eps, 5)
            self.spec = np.round(tn / (tn + fp + self.eps) + self.eps, 5)
            self.precis = np.round(tp / (tp + fp + self.eps) + self.eps, 5)
            self.bal_acc = np.round(np.mean([self.sens, self.spec]), 5)
            self.neg_pred_val = np.round(tn / (tn + fn + self.eps) + self.eps, 5)
            self.f1 = np.round(2 * self.sens * self.precis / (self.sens + self.precis + self.eps) + self.eps, 5)
            self.mcc = np.round((tp * tn - fp * fn) / np.sqrt(np.float64(tp + fp) * (fn + tn) * (tp + fn) * (fp + tn) + self.eps) + self.eps, 5)
            self.fnr = np.round(fn / (tp + fn + self.eps) + self.eps, 5)
            self.fpr = np.round(fp / (tn + fp + self.eps) + self.eps, 5)

        self.calibration_results = None
