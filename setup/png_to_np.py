import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image


def rasterize(in_file):
    img = np.array(Image.open(in_file))
    if np.sum(img) == 0:
        print('Zero array, {}'.format(os.path.basename(in_file)))
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert png to npy')
    parser.add_argument('--data_dir', type=str, default='../data/source/labels/')
    parser.add_argument('--out_basepath', type=str, default='../data/train/labels/')
    args = parser.parse_args()

    os.makedirs(args.out_basepath, exist_ok=True)

    for aoi in sorted(glob(os.path.join(args.data_dir, '*'))):
        print(aoi)
        for img in tqdm(sorted(glob(os.path.join(aoi, '*')))):
            # aoi_cube-t
            out_name = os.path.basename(img).split('.')[0]
            out_file = os.path.join(args.out_basepath, out_name)
            out_array = rasterize(img)
            if out_array is not None:
                np.save(out_file, out_array)


