import argparse
import itertools
import os
import numpy as np
import pandas as pd
import datetime as dt

from dateutil import relativedelta
from glob import glob
from tqdm import tqdm


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class ImageCrop:
    def __init__(self, img_dir, label_dir, cropsize, overlap_ratio=1., method='random',
            subset_file=None):
        np.random.seed(0)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.cropsize = int(cropsize)
        self.overlap_ratio = overlap_ratio
        self.method = method
        self.subset_file = subset_file

    def __call__(self, out_file='./train.csv'):
        if self.method == 'random':
            self.random(self.cropsize, self.img_dir, self.label_dir, out_file,
                    self.subset_file)
        elif self.method == 'sliding_window':
            self.sliding_window(self.cropsize, self.img_dir, out_file, self.subset_file,
                    self.overlap_ratio)
        else:
            print('Method is not implemented, try \'random\' or \'sliding_window\'')
            exit()

    @staticmethod
    def random(cropsize, img_dir, label_dir, out_file, subset_file=None, max_attempt=128):
        patches_per_img = (1024 // cropsize) ** 2
        incomplete = 0
        img_list = sorted(glob(os.path.join(img_dir, '*.npy')))
        # iterate around img_list
        img_list = [os.path.basename(i).split('.')[0] for i in img_list]
        aoi_list = [i.split('-')[0] for i in img_list]
        aoi_list = list(set(aoi_list))
        # assumes subset file has no duplicates
        if subset_file:
            with open(subset_file) as f:
                aoi_list = [l.replace('\n', '').replace('/', '_') for l in f]
            print(aoi_list)
        out_dict = {
                'old_file': [],
                'new_file': [],
                'label_file': [],
                'y_start': [],
                'x_start': [],
        }
        for cube_id in aoi_list:
            # dates are in chronological order due to sorting the glob
            date_list = ['-'.join(i.split('-')[1:]) for i in img_list if \
                    i.startswith(cube_id)]
            year_month_list = ['-'.join(i.split('-')[:2]) for i in date_list]
            year_month_list = sorted(list(set(year_month_list)))
            year_month_pair_list = pairwise(year_month_list)
            for start_year_month, end_year_month in tqdm(year_month_pair_list):
                start_date_list = [i for i in date_list if i.startswith(start_year_month)]
                end_date_list = [i for i in date_list if i.startswith(end_year_month)]

                start_file = '-'.join([cube_id, start_date_list[0]])
                end_file = '-'.join([cube_id, end_date_list[0]])
                label_file = '-'.join([cube_id, end_year_month, start_year_month])

                label = np.load(os.path.join(label_dir, '{}.npy'.format(label_file)))
                label[label > 0] = 1

                h = 1024
                w = 1024
                positive_patch_count = 0
                negative_patch_count = 0
                crop_attempt = 0
                while True:
                    y_start = np.random.randint(0, h-cropsize+1)
                    x_start = np.random.randint(0, w-cropsize+1)
                    label_crop = label[x_start:x_start+cropsize, y_start:y_start+cropsize]
                    if np.sum(label_crop) > 0:
                        if positive_patch_count < patches_per_img:
                            positive_patch_count += 1
                            out_dict['old_file'].append(start_file)
                            out_dict['new_file'].append(end_file)
                            out_dict['label_file'].append(label_file)
                            out_dict['y_start'].append(y_start)
                            out_dict['x_start'].append(x_start)
                    else:
                        if negative_patch_count < patches_per_img:
                            negative_patch_count += 1
                            out_dict['old_file'].append(start_file)
                            out_dict['new_file'].append(end_file)
                            out_dict['label_file'].append(label_file)
                            out_dict['y_start'].append(y_start)
                            out_dict['x_start'].append(x_start)
                    crop_attempt += 1
                    if positive_patch_count == patches_per_img and \
                            negative_patch_count == patches_per_img:
                        break
                    elif crop_attempt == max_attempt:
                        print('{}: {} positive and {} negative patches'.format(label_file,
                            positive_patch_count, negative_patch_count))
                        incomplete += 1
                        break
        print(incomplete, 'are incomplete')
        out_df = pd.DataFrame.from_dict(out_dict)
        out_df.to_csv(out_file, index=False)

    @staticmethod
    def sliding_window(cropsize, img_dir, out_file, subset_file=None, overlap_ratio=1.):
        img_list = sorted(glob(os.path.join(img_dir, '*.npy')))
        # iterate around img_list
        img_list = [os.path.basename(i).split('.')[0] for i in img_list]
        aoi_list = [i.split('-')[0] for i in img_list]
        aoi_list = list(set(aoi_list))
        # assumes subset file has no duplicates
        if subset_file:
            with open(subset_file) as f:
                aoi_list = [l.replace('\n', '').replace('/', '_') for l in f]
            print(aoi_list)
        out_dict = {
                'old_file': [],
                'new_file': [],
                'label_file': [],
                'y_start': [],
                'x_start': [],
        }
        # overlap 1 means no overlap
        spacing = int(overlap_ratio * cropsize)
        for cube_id in aoi_list:
            # dates are in chronological order due to sorting the glob
            date_list = ['-'.join(i.split('-')[1:]) for i in img_list if
                    i.startswith(cube_id)]
            year_month_list = ['-'.join(i.split('-')[:2]) for i in date_list]
            year_month_list = sorted(list(set(year_month_list)))
            year_month_pair_list = pairwise(year_month_list)
            for start_year_month, end_year_month in year_month_pair_list:
                start_date_list = [i for i in date_list if i.startswith(start_year_month)]
                end_date_list = [i for i in date_list if i.startswith(end_year_month)]

                start_file = '-'.join([cube_id, start_date_list[0]])
                end_file = '-'.join([cube_id, end_date_list[0]])
                label_file = '-'.join([cube_id, end_year_month, start_year_month])
                h = 1024
                w = 1024
                # ignores final incomplete patches
                for y_start in range(0, h-cropsize+1, spacing):
                    for x_start in range(0, w-cropsize+1, spacing):
                        out_dict['old_file'].append(start_file)
                        out_dict['new_file'].append(end_file)
                        out_dict['label_file'].append(label_file)
                        out_dict['y_start'].append(y_start)
                        out_dict['x_start'].append(x_start)
        out_df = pd.DataFrame.from_dict(out_dict)
        out_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create csv from cropping labels')
    parser.add_argument('--img_dir', type=str, default='../data/train/images/')
    parser.add_argument('--label_dir', type=str, default='../data/train/labels/')
    parser.add_argument('--cropsize', type=int, default=128)
    parser.add_argument('--method', type=str, choices=['random', 'sliding_window'],
            default='random')
    parser.add_argument('--subset_file', type=str,
            default='../data/source/train_with_labels.txt')
    args = parser.parse_args()

    crop = ImageCrop(
            img_dir=args.img_dir,
            label_dir=args.label_dir,
            cropsize=args.cropsize,
            method=args.method,
            subset_file=args.subset_file,
    )
    crop()

