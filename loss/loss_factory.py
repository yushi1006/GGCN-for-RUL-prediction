from torch import nn
import torch
import numpy as np


def QuantileLoss(outputs, targets, quantile_level):
    index = (outputs <= targets).float()
    loss = quantile_level * (targets - outputs) * index + (1 - quantile_level) * (outputs - targets) * (1 - index)
    return loss.mean()
