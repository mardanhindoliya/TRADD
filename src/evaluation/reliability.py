import numpy as np
import torch


def get_probabilities(model, loader, device):
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(device))
            probs.extend(
                torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            )
            labels.extend(y.numpy())

    return np.array(probs), np.array(labels)


def find_threshold(probs, labels, target_class, min_precision=0.90):
    candidates = np.sort(np.unique(probs))

    best_threshold = None
    best_coverage = 0

    for threshold in candidates:
        if target_class == 0:
            accepted = probs <= threshold
        else:
            accepted = probs >= threshold

        if accepted.sum() < 5:
            continue

        precision = np.mean(labels[accepted] == target_class)

        if precision >= min_precision and accepted.sum() > best_coverage:
            best_threshold = threshold
            best_coverage = accepted.sum()

    return best_threshold
