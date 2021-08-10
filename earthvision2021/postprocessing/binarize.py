import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarize predictions to one channel')
    parser.add_argument('--in_dir', type=str, default='../results/test/')
    args = parser.parse_args()

    out_dir = os.path.join(args.in_dir, 'binary')
    os.makedirs(out_dir, exist_ok=True)

    for img_file in tqdm(sorted(glob(os.path.join(args.in_dir, '*.npy')))):
        file_name = os.path.basename(img_file).split('.')[0]
        array = np.load(img_file)
        array = np.argmax(array, axis=0).astype(np.uint8)

        pil_image = Image.fromarray(array)
        pil_image.save(os.path.join(out_dir, file_name + '.png'))

