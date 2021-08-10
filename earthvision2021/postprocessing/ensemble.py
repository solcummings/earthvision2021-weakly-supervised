import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble predictions by averaging')
    parser.add_argument('--in_dir_list', type=list, nargs='+')
    parser.add_argument('--out_dir', type=str, default='../results/ensemble/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # assumes all prediction directories have the same output filenames
    in_dir = args.in_dir_list[0]
    img_list = sorted(glob(os.path.join(in_dir, '*.npy')))
    img_list = [os.path.basename(i) for i in img_list]
    for img_file in tqdm(img_list):
        array_list = [np.load(os.path.join(i, img_file)) for i in args.in_dir_list]
        # average results
        out_array = np.average(np.array(array_list), axis=0)
        np.save(os.path.join(args.out_dir, img_file.split('.')[0]), out_array)

