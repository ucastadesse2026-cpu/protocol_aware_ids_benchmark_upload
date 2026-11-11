"""Detection metrics. Positive class = attack (1). FNR = missed attacks; FPR = false alarms."""
from __future__ import annotations

from typing import Dict

import numpy as np


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else float("nan")
    return dict(acc=(tp + tn) / max(len(y), 1), precision=prec, recall=rec, f1=f1,
                fpr=div(fp, fp + tn), fnr=div(fn, fn + tp))


def evaluate(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return classification_metrics(y, model.predict_proba(X).argmax(axis=1))
