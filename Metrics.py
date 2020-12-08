import torch
import numpy as np
import torch.nn.functional as F
def accuracy_from_scores(scores, target):
    """
    Given the unnormalized scores and the gold standard labels, return the average accuracy
    Currently, there is no support for classes to be ignored

    Args:
        inp(torch.Tensor or np.array): Predicted labels by the model
        target(torch.Tensor or np.array): Gold standard labels
    """
    preds = scores_to_class(scores)
    return accuracy(preds, target)
def accuracy(inp, target):
    """
    Given predicted classes and the gold standard labels, return the average accuracy
    Currently, there is no support for classes to be ignored

    Args:
        inp(torch.Tensor or np.array): Predicted labels by the model
        target(torch.Tensor or np.array): Gold standard labels
    """
    accs = torch.zeros(target.shape)
    accs[inp == target] = 1
    return float(torch.mean(accs))
def scores_to_class(scores):
    """
    Given un-normalized scores, convert them to a probability distribution using the
    softmax function

    Args:
        scores(torch.tensor): Tensor of N x K where N is the batch size and K is the number of classes
    """
    return torch.argmax(F.softmax(scores, dim=1), dim=1)