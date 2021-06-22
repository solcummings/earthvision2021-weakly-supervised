import argparse
import os
import datetime as dt
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from common import load_yaml

from utils import datasets
from utils import misc
from utils import transforms
import models


class Predicting:
    def __init__(self, config_path, disable_tqdm=False):
        self.config = load_yaml.load(config_path)
        self.disable_tqdm = disable_tqdm

        misc.seeds.set_seeds(self.config['seed'], self.config['deterministic'])

        self.amp = self.config['amp']
        self.dataloader = datasets.build('test', self.config['dataset_args'])

        self.tta = True if len(self.config['dataset_args']['transforms_args'].keys()) \
                > 1 else False
        if self.tta:
            self.transforms = transforms.tta.TTA(
                    self.config['dataset_args']['transforms_args'], debug=False)

        self.device = torch.device('cuda')
        self.model = models.build(**self.config)
        self.model = self.model.to(self.device)
        self.softmax = torch.nn.Softmax(dim=1).to(self.device)

        self.classes = self.config['model_args']['classes']

        self.save_dir = self.config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

    def predict(self):
        print('--- Starting Predicting ---')
        self.model.eval()
        with torch.no_grad():
            self.iterate_dataloader(self.dataloader)

    def iterate_dataloader(self, dataloader):
        for batch in tqdm(dataloader, disable=self.disable_tqdm):
            img1, img2, old_name_list, new_name_list = batch
            if self.tta:
                # tta returns list for each pattern of list of tensors
                img_pattern = self.transforms([img1, img2])
                total_prediction = []
                for augmented_img1, augmented_img2 in img_pattern:
                    augmented_img1 = augmented_img1.to(self.device, dtype=torch.float, non_blocking=True)
                    augmented_img2 = augmented_img2.to(self.device, dtype=torch.float, non_blocking=True)
                    prediction = self.softmax(self.__autocast_prediction(augmented_img1, augmented_img2))
                    prediction = prediction.to('cpu').numpy()
                    total_prediction.append(prediction)
                total_prediction = self.transforms.reverse(total_prediction)
                # mean tta outputs
                prediction = np.mean(total_prediction, axis=0)
            else:
                img1 = img1.to(self.device, dtype=torch.float, non_blocking=True)
                img2 = img2.to(self.device, dtype=torch.float, non_blocking=True)
                prediction = self.softmax(self.__autocast_prediction(img1, img2))
                prediction = prediction.to('cpu').numpy()
            # (batch, classes)
            for old_name, new_name, save_array in zip(
                    old_name_list, new_name_list, prediction):
                if self.config['dataset_args']['comparison'][:7] == 'monthly':
                    old_date = dt.date(
                            year=int(old_name.split('-')[1]),
                            month=int(old_name.split('-')[2]),
                            day=1,
                    )
                    new_date = dt.date(
                            year=int(new_name.split('-')[1]),
                            month=int(new_name.split('-')[2]),
                            day=1,
                    )
                    old_date_str = old_date.strftime('%Y-%m')
                    new_date_str = new_date.strftime('%Y-%m')
                save_name = '-'.join([old_name.split('-')[0], new_date_str, old_date_str])
                np.save(os.path.join(self.save_dir, save_name), save_array)

    def __autocast_prediction(self, *args, **kwargs):
        if self.amp:
            with torch.cuda.amp.autocast():
                prediction = self.model(*args)
        else:
            prediction = self.model(*args)
        return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='./config_predict.yml')
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()

    model = Predicting(args.config, args.disable_tqdm)
    model.predict()


