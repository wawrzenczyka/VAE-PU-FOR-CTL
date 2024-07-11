import numpy as np
from sklearn import metrics


def calculate_metrics(
    y_proba,
    y,
    ls_s_proba,
    no_ls_s_proba,
    method=None,
    ls_pi=None,
    time=None,
    augmented_label_shift=False,
):
    y = np.where(y == 1, 1, 0)

    if augmented_label_shift:
        y_pred = np.where(y_proba * ls_s_proba / no_ls_s_proba > 0.5, 1, 0)
    else:
        y_pred = np.where(y_proba > 0.5, 1, 0)

    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)
    auc = metrics.roc_auc_score(y, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y, y_pred)

    metric_values = {
        "Method": method,
        "Label shift \\pi": ls_pi,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1,
        "AUC": auc,
        "Balanced accuracy": balanced_accuracy,
        "Time": time,
    }
    return metric_values
