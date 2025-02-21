import torch
import numpy as np
import random
import torch.nn.functional as F


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(labels, labels_te, y, y_te):
    train_error = 1.0 - labels.eq(y).float().mean().item()
    test_error = 1.0 - labels_te.eq(y_te).float().mean().item()
    return train_error, test_error
