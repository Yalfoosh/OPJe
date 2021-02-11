import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader


def get_labels(model, dataset: Dataset, batch_size: int, collate_function=None):
    y_true = list()
    y_pred = list()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_function,
    )

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch

            y_true.extend(labels)
            y_pred.extend(torch.argmax(model(inputs), dim=1))

    return y_true, y_pred


def get_metrics(confusion_matrix: np.ndarray):
    tp = int(np.diag(confusion_matrix)[0])
    fp = int((confusion_matrix.sum(axis=0) - tp)[0])
    fn = int((confusion_matrix.sum(axis=1) - tp)[0])
    tn = int(confusion_matrix.sum() - (tp + fp + fn))

    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    numerator = 2 * precision * recall
    denominator = precision + recall

    if abs(denominator) < 1e-6:
        denominator = 1.0

    f1 = numerator / denominator

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model, dataset: Dataset, batch_size: int, collate_function: None):
    model.eval()

    y_true, y_pred = get_labels(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        collate_function=collate_function,
    )

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    metrics = get_metrics(confusion_matrix=confusion_matrix)

    return metrics
