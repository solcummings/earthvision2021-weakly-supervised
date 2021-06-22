import logging
logger = logging.getLogger(__name__)
import os
import torchvision
import numpy as np
import pandas as pd

from utils.transforms.compose import ComposeTransforms


class Dataset:
    def __init__(self, phase, img_dir, mean_std, crop_file, cropsize, aoi_file,
            transforms_args, **kwargs):
        self.phase = phase
        self.img_dir = img_dir
        self.crop_file = crop_file
        self.cropsize = cropsize
        self.aoi_file = aoi_file
        self.df = self.load_data()
        logger.info('{} has {} patches'.format(self.phase.title(), len(self.df)))

        crop_args = transforms_args.pop('centercrop')
        self.patchsize = crop_args['patchsize']
        self.transforms = ComposeTransforms(
                transforms_dict={'centercrop': crop_args},
                mean_std=mean_std,
                to_tensor=False,
        )

    def __getitem__(self, index):
        old_name = self.df.iloc[index]['old_file']
        new_name = self.df.iloc[index]['new_file']
        x_start = self.df.iloc[index]['x_start']
        y_start = self.df.iloc[index]['y_start']
        old_img = np.load(os.path.join(self.img_dir, old_name + '.npy'), mmap_mode='r')
        new_img = np.load(os.path.join(self.img_dir, new_name + '.npy'), mmap_mode='r')

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
        # transforms return list of array
        old_img, new_img = self.transforms([old_img, new_img], [old_name, new_name])
        return old_img, new_img, old_name, new_name

    def load_data(self):
        return pd.read_csv(self.crop_file)

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    pass

