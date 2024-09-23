import os
import torch
import random
from collections import OrderedDict
from torchvision.utils import make_grid
from abc import ABC, abstractmethod
# Define the legal metrics here
LEGAL_METRIC = ['Acc', 'AUC', 'LogLoss']

class AbstractTrainer(object):
    def __init__(self, config, stage="Train"):
        feasible_stage = ["Train", "Test"]
        if stage not in feasible_stage:
            raise ValueError(f"stage should be in {feasible_stage}, but found '{stage}'")

        self.config = config
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        config_cfg = config.get("config", {})

        self.model_name = model_cfg.get("name", "model")

        self.gpu = None
        self.dir = None
        self.debug = None
        self.device = None
        self.resume = None
        self.local_rank = None
        self.num_classes = model_cfg.get("num_classes", 1)

        self.best_metric = 0.0
        self.best_step = 1
        self.start_step = 1

        self._initiated_settings(model_cfg, data_cfg, config_cfg)

        if stage == 'Train':
            self._train_settings(model_cfg, data_cfg, config_cfg)
        if stage == 'Test':
            self._test_settings(model_cfg, data_cfg, config_cfg)

    @abstractmethod
    def _initiated_settings(self, model_cfg, data_cfg, config_cfg):
        pass

    @abstractmethod
    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        pass

    @abstractmethod
    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        pass

    @abstractmethod
    def _save_ckpt(self, step, best=False):
        pass

    @abstractmethod
    def _load_ckpt(self, best=False, train=False):
        pass

    def to_device(self, items):
        if isinstance(items, (list, tuple)):
            return tuple(self.to_device(obj) for obj in items)
        elif isinstance(items, torch.Tensor):
            return items.to(self.device)
        else:
            return items  # Return as-is for non-tensor objects

    @staticmethod
    def fixed_randomness():
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self, epoch, step):
        pass

    @abstractmethod
    def test(self):
        pass

    def plot_figure(self, images, pred, gt, nrow, categories=None, show=True):
        import matplotlib.pyplot as plt
        plot = make_grid(
            images, nrow, padding=4, normalize=True, scale_each=True, pad_value=1)
        if self.num_classes == 1:
            pred = (pred >= 0.5).cpu().numpy()
        else:
            pred = pred.argmax(1).cpu().numpy()
        gt = gt.cpu().numpy()
        if categories is not None:
            pred = [categories[i] for i in pred]
            gt = [categories[i] for i in gt]
        plot = plot.permute([1, 2, 0])
        plot = plot.cpu().numpy()
        ret = plt.figure()
        plt.imshow(plot)
        plt.title("pred: %s\ngt: %s" % (pred, gt))
        plt.axis("off")
        if show:
            plt.savefig(os.path.join(self.dir, "test_image.png"), dpi=300)
            plt.show()
            plt.close()
        else:
            plt.close()
            return ret
