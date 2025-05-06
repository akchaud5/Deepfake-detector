import os
import torch
import glob
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer.abstract_trainer import AbstractTrainer
from model.network import Recce  # Assuming your model is called Recce and is located in model/network.py
from dataset import CelebDF
from torch.utils.data import DataLoader
from trainer.utils import AverageMeter, AccMeter, AUCMeter
from torch.utils.tensorboard import SummaryWriter
import yaml

class ExpMultiGpuTrainer(AbstractTrainer):
    def _initiated_settings(self, model_cfg, data_cfg, config_cfg):
        # Directories for checkpoints and logs can be overridden in the config
        self.checkpoint_dir = config_cfg.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        log_dir = config_cfg.get('log_dir', os.path.join(self.checkpoint_dir, 'tensorboard_logs'))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.debug = config_cfg.get('debug', False)
        self.resume = config_cfg.get('resume', False)
        self.device = torch.device(config_cfg.get('device', 'cpu'))
        self.local_rank = config_cfg.get('local_rank', 0)

        self.train_loader = self._get_data_loader(data_cfg, branch=data_cfg['train_branch'], stage='train')
        self.val_loader = self._get_data_loader(data_cfg, branch=data_cfg['val_branch'], stage='val')
        self.log_steps = config_cfg.get('log_steps', 100)

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        model_cfg = {k: v for k, v in model_cfg.items() if k != 'name'}
        self.model = self.load_model(self.model_name)(**model_cfg).to(self.device)

        if torch.cuda.is_available() and config_cfg['device'] == 'cuda':
            if torch.distributed.is_initialized():
                self.model = DDP(self.model, device_ids=[self.local_rank])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_cfg['optimizer']['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config_cfg['scheduler']['step_size'], gamma=config_cfg['scheduler']['gamma'])

        self.start_step = 0
        self.start_epoch = 0
        self.best_metric = 0
        if self.resume:
            self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        model_cfg = {k: v for k, v in model_cfg.items() if k != 'name'}
        self.model = self.load_model(self.model_name)(**model_cfg).to(self.device)
        self._load_ckpt(best=True)

    def _save_ckpt(self, step, epoch, best=False):
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'step': step,
            'epoch': epoch,
            'best_metric': self.best_metric
        }
        filename = os.path.join(self.checkpoint_dir, f'step_{step}_model.pt' if not best else 'best_model.pt')
        torch.save(state, filename)

    def _load_ckpt(self, best=False, train=False):
        if not best:
            filename = self.config.get('ckpt', '')  # Use the checkpoint path from config if not best
            if filename:
                checkpoint_file = filename
            else:
                checkpoint_file = max(glob.glob(os.path.join(self.checkpoint_dir, '*.pt')), key=os.path.getctime, default=None)
        else:
            checkpoint_file = os.path.join(self.checkpoint_dir, 'best_model.pt')

        if not checkpoint_file or not os.path.exists(checkpoint_file):
            if train:
                print(f"No checkpoint found at {checkpoint_file}. Starting training from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])

        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            else:
                print("Warning: 'scheduler_state' not found in checkpoint. Scheduler will start from scratch.")
            
            self.start_step = checkpoint['step'] + 1
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint['best_metric']

        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch} and step {self.start_step}.")

    def train(self):
        print("********** Training begins...... **********")
        loss_meter = AverageMeter()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()
        steps_per_epoch = len(self.train_loader)
        total_epochs = 10  # Fixed number of epochs
        target_step = total_epochs * steps_per_epoch

        if self.start_step >= target_step:
            print(f"Training already completed. Target steps ({target_step}) reached.")
            return

        global_step = self.start_step  # Initialize global_step before the loop

        for epoch in range(self.start_epoch, total_epochs):
            if epoch * steps_per_epoch + self.start_step >= target_step:
                print(f"Training already completed up to epoch {epoch}.")
                break  # Exit if training is already complete

            self.model.train()
            auc_meter.reset()
            loss_meter.reset()
            acc_meter.reset()

            print(f"Starting epoch {epoch+1}...")  # Print start of epoch

            for step, (I, Y) in enumerate(self.train_loader):
                global_step = epoch * steps_per_epoch + step  # Update global_step within the loop

                # Skip steps until reaching start_step
                if global_step < self.start_step:
                    continue

                in_I, Y = self.to_device((I, Y))
                Y = Y.long()

                Y_pre = self.model(in_I)
                loss = self.criterion(Y_pre, Y)
                loss_meter.update(loss.item(), I.size(0))  # Corrected line

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc_meter.update(Y_pre, Y)
                auc_meter.update(Y_pre, Y)

                if global_step % self.log_steps == 0:  # Update this line
                    self.writer.add_scalar('Loss/train', loss_meter.avg, global_step)
                    self.writer.add_scalar('Accuracy/train', acc_meter.mean_acc(), global_step)
                    self.writer.add_scalar('AUC/train', auc_meter.compute_auc(), global_step)

                    print(f"Epoch [{epoch+1}/{total_epochs}], Step [{step}/{steps_per_epoch}], "
                          f"Global Step [{global_step}], "
                          f"Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.mean_acc():.4f}, AUC: {auc_meter.compute_auc():.4f}")

                if global_step % 1000 == 0:
                    self._save_ckpt(global_step, epoch)

            # Validation and checkpointing at the end of the epoch
            self.scheduler.step()  # Correct order: scheduler.step() should follow optimizer.step()
            print(f"Epoch {epoch+1}/{total_epochs} finished. Starting validation...")
            self.validate(epoch, global_step)
            print(f"Validation for Epoch {epoch+1} completed.")
            self._save_ckpt(global_step, epoch)  # Save the checkpoint at the end of each epoch

            # Reset start_step after completing an epoch
            self.start_step = 0

    def validate(self, epoch, step):
        print("********** Validation begins...... **********")
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()
        auc_meter.reset()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for val_step, (I, Y) in enumerate(self.val_loader, 1):
                in_I, Y = self.to_device((I, Y))
                Y = Y.long()
                Y_pre = self.model(in_I)
                loss = self.criterion(Y_pre, Y)
                loss_meter.update(loss.item(), I.size(0))
                acc_meter.update(Y_pre, Y)
                auc_meter.update(Y_pre, Y)

                all_preds.append(Y_pre.cpu())
                all_labels.append(Y.cpu())

                self.writer.add_scalar('Loss/val', loss_meter.avg, step + val_step)
                self.writer.add_scalar('Accuracy/val', acc_meter.mean_acc(), step + val_step)
                self.writer.add_scalar('AUC/val', auc_meter.compute_auc(), step + val_step)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        print(f"Validation: Epoch [{epoch}], Step [{step}]")
        print(f"Predictions: {all_preds[:10]}")
        print(f"Labels: {all_labels[:10]}")
        print(f"Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.mean_acc():.4f}, AUC: {auc_meter.compute_auc():.4f}")

        if acc_meter.mean_acc() > self.best_metric:
            self.best_metric = acc_meter.mean_acc()
            self.best_step = epoch
            self._save_ckpt(step, epoch, best=True)

    def load_model(self, model_name):
        models = {
            'Recce': Recce
        }
        return models[model_name]

    def test(self):
        print("********** Testing begins...... **********")
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()

        with torch.no_grad():
            for test_step, (I, Y) in enumerate(self.test_loader, 1):
                in_I, Y = self.to_device((I, Y))
                Y_pre = self.model(in_I)
                loss = self.criterion(Y_pre, Y)
                loss_meter.update(loss.item(), I.size(0))
                acc_meter.update(Y_pre, Y)
                auc_meter.update(Y_pre, Y)

        print(f"Test: Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.mean_acc():.4f}, AUC: {auc_meter.compute_auc():.4f}")

    def _get_data_loader(self, cfg, branch, stage):
        with open(cfg['file'], 'r') as file:
            data_cfg = yaml.safe_load(file)

        branch_cfg = data_cfg[branch]
        branch_cfg['split'] = stage  # Set the correct split

        print(f"Loading data for {stage}...")  # Indicate data loading start
        dataset = CelebDF(branch_cfg)
        shuffle = stage == 'train'
        batch_size = cfg[f'{stage}_batch_size']
        num_workers = cfg.get('num_workers', 1)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        print(f"Data for {stage} loaded. Batch size: {batch_size}")  # Confirm data loaded
        return loader
