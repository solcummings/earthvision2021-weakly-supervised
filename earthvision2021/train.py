import argparse
import logging
logger = logging.getLogger(__name__)
import os
import shutil
import torch

from tqdm import tqdm

from common import load_yaml

from utils import criteria
from utils import datasets
from utils import misc
from utils import optimizers
from utils import schedulers
import models


class Training:
    def __init__(self, config_path, disable_tqdm=False):
        self.config = load_yaml.load(config_path)
        self.disable_tqdm = disable_tqdm

        misc.seeds.set_seeds(self.config['seed'], self.config['deterministic'])
        self.amp = self.config['amp']
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.train_dataloader = datasets.build('train', self.config['train_dataset_args'])
        self.val_dataloader = datasets.build('val', self.config['val_dataset_args'])

        self.device = torch.device('cuda')
        self.model = models.build(**self.config)
        self.model = self.model.to(self.device)
        logger.info(self.model)

        self.criterion = criteria.build(**self.config)
        self.criterion = self.criterion.to(self.device)

        self.optimizer = optimizers.build(self.model.parameters(), **self.config)
        self.scheduler = schedulers.build(self.optimizer, **self.config)

        self.save_dir = self.config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        log_keys = ['epoch',
                'train_loss', 'train_iou', 'train_recall', 'train_precision', 'train_f1',
                'val_loss', 'val_iou', 'val_recall', 'val_precision', 'val_f1']
        self.log = misc.log.TrainLog(
                self.save_dir,
                log_keys,
                save_keys=['val_loss', 'val_iou', 'val_f1'],
                save_modes=['min', 'max', 'max'],
        )

        self.start_epoch = 1
        self.final_epoch = self.config['epochs']
        if self.config['load_checkpoint']:
            self.__load_checkpoint()
        shutil.copy(config_path, os.path.join(self.save_dir, 'config.yml'))

    def train(self):
        logger.info('--- Starting Training ---')
        for epoch in range(self.start_epoch, self.final_epoch + 1):
            logger.info('[Epoch {}]'.format(epoch))
            self.model.train()
            train_loss, train_metrics_dict = \
                    self.iterate_dataloader('train', self.train_dataloader)
            self.model.eval()
            with torch.no_grad():
                val_loss, val_metrics_dict = \
                        self.iterate_dataloader('val', self.val_dataloader)

                scheduler_update_basis = val_loss if isinstance(self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau) else None
                self.scheduler.step(scheduler_update_basis)

                epoch_dict = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                } | train_metrics_dict | val_metrics_dict
                checkpoint_dict = {
                        'epoch': epoch,
                        'model': self.model,
                        'optimizer': self.optimizer,
                        'scheduler': self.scheduler,
                }
                # log and update best metrics
                checkpoint_dict, filename_list = self.log(checkpoint_dict, epoch_dict)
                if filename_list != []:
                    for filename in filename_list:
                        models.model_base.save_checkpoint(
                                os.path.join(self.save_dir, filename),
                                **checkpoint_dict,
                        )
                if epoch % 20 == 0:
                    filename = 'epoch_{}.pth'.format(epoch)
                    models.model_base.save_checkpoint(
                            os.path.join(self.save_dir, 'checkpoints', filename),
                            **checkpoint_dict,
                    )

    def iterate_dataloader(self, phase, dataloader):
        mean_loss = 0.  # mean for epoch
        metric = criteria.metric.PrecisionRecall(phase,
                self.config['model_args']['classes'])
        for batch in tqdm(dataloader, disable=self.disable_tqdm):
            img1, img2, label = batch
            img1 = img1.to(self.device, dtype=torch.float, non_blocking=True)
            img2 = img2.to(self.device, dtype=torch.float, non_blocking=True)
            label = label.to(self.device, dtype=torch.long, non_blocking=True)

            if phase == 'train':
                for p in self.model.parameters():
                    p.grad = None  # faster than optimizer.zero_grad or model.zero_grad

            if self.amp:
                with torch.cuda.amp.autocast():
                    prediction = self.model(img1, img2)
                    loss = self.criterion(prediction, label)
            else:
                prediction = self.model(img1, img2)
                loss = self.criterion(prediction, label)

            mean_loss += loss.item()
            prediction = torch.max(prediction, 1)[1]  # second output is prediction tensor
            prediction = prediction.to('cpu')
            label = label.to('cpu')
            metric(prediction, label)

            if phase == 'train':
                if self.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

        mean_loss = mean_loss / len(dataloader)
        metric_dict = metric.calculate()
        return mean_loss, metric_dict

    def __load_checkpoint(self):
        checkpoint_dict = models.model_base.load_checkpoint(
                self.config['load_checkpoint'])
        self.start_epoch = checkpoint_dict['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        self.log.best_values = checkpoint_dict['best_values']
        self.log.update_count = checkpoint_dict['update_count']
        logger.info('--- Resuming from {} ---'.format(self.config['load_checkpoint']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='./config_train.yml')
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()

    model = Training(args.config, args.disable_tqdm)
    model.train()


