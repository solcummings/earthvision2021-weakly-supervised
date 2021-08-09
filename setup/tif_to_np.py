import argparse
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from osgeo import gdal


def rasterize(in_file):
    img = gdal.Open(in_file).ReadAsArray()
    if np.sum(img) == 0:
        print('Zero array, {}'.format(os.path.basename(in_file)))
        return None
    else:
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tif to npy')
    parser.add_argument('--data_dir', type=str, default='../data/source/planet_val/')
    parser.add_argument('--out_basepath', type=str, default='../data/train/images_val')
    args = parser.parse_args()

    os.makedirs(args.out_basepath, exist_ok=True)
    mean_std_dict = {'img_name': [], 'mean': [], 'std': []}

    for aoi in sorted(glob(os.path.join(args.data_dir, '*'))):
        print(aoi)
        for cube_id in sorted(glob(os.path.join(aoi, '*'))):
            aoi_cube_dir = os.path.join(cube_id, 'L3H-SR')
            for img in tqdm(sorted(glob(os.path.join(aoi_cube_dir, '*')))):
                # aoi_cube-t
                out_name = os.path.basename(aoi) + '_' + os.path.basename(cube_id) + \
                        '-' + os.path.basename(img).split('.')[0]
                out_file = os.path.join(args.out_basepath, out_name)
                out_array = rasterize(img)
                if out_array is not None:
                    np.save(out_file, out_array)

                    # mean and array
                    mean = list(np.mean(out_array, axis=(1,2)))
                    std = list(np.std(out_array, axis=(1,2)))
                    mean_std_dict['img_name'].append(out_name)
                    mean_std_dict['mean'].append(mean)
                    mean_std_dict['std'].append(std)

    out_df = pd.DataFrame.from_dict(mean_std_dict)
    out_df.to_csv('./mean_std.csv', index=False)


