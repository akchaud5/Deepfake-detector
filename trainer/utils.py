import torch
import os
import sys
import time
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import torch.distributed as dist

LEGAL_METRIC = ['Acc', 'AUC', 'LogLoss']

def exp_recons_loss(recons, x):
    x, y = x
    loss = torch.tensor(0., device=y.device)
    real_index = torch.where(1 - y)[0]
    for r in recons:
        if real_index.numel() > 0:
            real_x = torch.index_select(x, dim=0, index=real_index)
            real_rec = torch.index_select(r, dim=0, index=real_index)
            real_rec = F.interpolate(real_rec, size=x.shape[-2:], mode='bilinear', align_corners=True)
            loss += torch.mean(torch.abs(real_rec - real_x))
    return loss

def setup_for_distributed(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.distributed.barrier()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def center_print(content, around='*', repeat_around=10):
    num = repeat_around
    s = around
    print(num * s + ' %s ' % content + num * s)

def reduce_tensor(t):
    if not dist.is_initialized():
        return t
    rt = t.clone()
    dist.all_reduce(rt)
    rt /= float(dist.get_world_size())
    return rt

def tensor2image(tensor):
    image = tensor.permute([1, 2, 0]).cpu().detach().numpy()
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def state_dict(state_dict):
    """ Remove 'module' keyword in state dictionary. """
    weights = OrderedDict()
    for k, v in state_dict.items():
        weights.update({k.replace("module.", ""): v})
    return weights

def Timer():
    """
    Timer utility for measuring time intervals.
    """
    from timeit import default_timer as timer
    start = timer()
    def elapsed():
        return timer() - start
    return elapsed

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, targets):
        if predictions.dim() > 1:  # If predictions are in one-hot format
            predictions = torch.argmax(predictions, dim=1)
        if targets.dim() > 1:  # If targets are in one-hot format
            targets = torch.argmax(targets, dim=1)
        self.correct += (predictions == targets).sum().item()
        self.total += targets.size(0)

    def mean_acc(self):
        return self.correct / self.total if self.total > 0 else 0

from sklearn.metrics import roc_auc_score
import numpy as np

class AUCMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.targets = []
        self.scores = []

    def update(self, predictions, targets):
        if predictions.ndim > 1:
            # Assume that the predictions are logits, need to be converted to probabilities
            predictions = predictions.softmax(dim=1)[:, 1]

        self.scores.extend(predictions.cpu().detach().numpy())
        self.targets.extend(targets.cpu().numpy().tolist())

    def compute_auc(self):
        if len(set(self.targets)) == 1:  # If there's only one class in targets
            return 0.5
        return roc_auc_score(self.targets, self.scores)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def center_print(text, width=60, fill='*'):
    """
    Print text centered with a fill character around it.
    """
    print(text.center(width, fill))
