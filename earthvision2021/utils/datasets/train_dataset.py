import logging
logger = logging.getLogger(__name__)
import os
import random
import torchvision
import numpy as np
import pandas as pd

from utils.transforms.compose import ComposeTransforms


class Dataset:
    def __init__(self, phase, img_dir, label_dir, mean_std, crop_file, cropsize,
            transforms_args, val_ratio=0.2, aoi_file=None, seed=0, **kwargs):
        self.phase = phase
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.crop_file = crop_file
        self.cropsize = cropsize
        self.val_ratio = val_ratio
        self.aoi_file = aoi_file
        self.seed = seed

        self.df = self.load_data()
        logger.info('{} has {} patches'.format(self.phase.title(), len(self.df)))
        self.transforms = ComposeTransforms(
                transforms_dict=transforms_args,
                mean_std=mean_std,
                #debug=True,
        )

    def __getitem__(self, index):
        old_name = self.df.iloc[index]['old_file']
        new_name = self.df.iloc[index]['new_file']
        label_name = self.df.iloc[index]['label_file']
        x_start = self.df.iloc[index]['x_start']
        y_start = self.df.iloc[index]['y_start']
        old_img = np.load(os.path.join(self.img_dir, old_name + '.npy'), mmap_mode='r')
        new_img = np.load(os.path.join(self.img_dir, new_name + '.npy'), mmap_mode='r')
        label = np.load(os.path.join(self.label_dir, label_name + '.npy'), mmap_mode='r')

        old_img = old_img[
                :,
                x_start: x_start + self.cropsize,
                y_start: y_start + self.cropsize,
        ]
        new_img = new_img[
                :,
                x_start: x_start + self.cropsize,
                y_start: y_start + self.cropsize,
        ]
        label = label[
                ...,
                x_start: x_start + self.cropsize,
                y_start: y_start + self.cropsize,
        ]
        # transforms return list of array
        old_img, new_img, label = self.transforms(
                x=[old_img, new_img, label],
                name=[old_name, new_name],
        )
        return old_img, new_img, label

    def load_data(self) -> np.ndarray:
        df = pd.read_csv(self.crop_file)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        val_index = int(np.ceil(len(df) * self.val_ratio))

        if self.aoi_file:
            with open(self.aoi_file, 'r') as aoi_list:
                val_aoi = [aoi for aoi in aoi_list]
            val_aoi = [v.replace('/', '_').replace('\n','') for v in val_aoi]
            above_df = df[df['old_file'].str.startswith(tuple(val_aoi))]
            below_df = df[~df['old_file'].str.startswith(tuple(val_aoi))]
            df = pd.concat([above_df, below_df])

        if self.phase == 'train':
            df = df.iloc[val_index:]
        else:
            df = df.iloc[:val_index]
        return df

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    pass

