import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarize for hard pseudo labels')
    parser.add_argument('--in_dir', type=str, default='../results/test/')
    parser.add_argument('--out_dir', type=str, default='../data/train/labels/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for img_file in tqdm(sorted(glob(os.path.join(args.in_dir, '*.npy')))):
        file_name = os.path.basename(img_file).split('.')[0]
        array = np.load(img_file)
        array = np.argmax(array, axis=0).astype(np.uint8)
        np.save(os.path.join(args.out_dir, file_name), array)

