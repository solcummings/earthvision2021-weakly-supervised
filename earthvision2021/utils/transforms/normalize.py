import os
import numpy as np
import pandas as pd

from utils.transforms.transform_base import Transform


class Normalize(Transform):
    # only normalizes arrays with 3 dimensions
    # probability should be binary
    def __init__(self, mean_std, p=1., *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        if isinstance(mean_std, str):
            assert os.path.exists(mean_std) and mean_std.split('.')[-1] == 'csv'
            self.stats = pd.read_csv(mean_std)
            self.load_mean_std = self._from_df
            # check if mean is object (list/str/dict for input with multiple bands)
            self.object = True if self.stats['mean'].dtype == 'O' else False
        elif isinstance(mean_std, dict):
            assert all([i in mean_std for i in ['mean', 'std']])
            self.stats = mean_std
            self.load_mean_std = self._from_dict
        else:
            print('--> No normalization')
            assert self.disable

    def forward(self, array_list, filename):
        if isinstance(filename, list):
            output = []
            for i, array in enumerate(array_list):
                # normalize first 2 inputs
                if i <= 1:
                    mean, std = self.load_mean_std(filename[i])
                    output.append(self.normalize(array, mean, std))
                else:
                    output.append(array)
            return output
        else:
            mean, std = self.load_mean_std(filename)
            return [self.normalize(array, mean, std) if array.ndim >= 3 else
                    array for array in array_list]

    def backward(self, array_list, filename):
        mean, std = self.load_mean_std(filename)
        return [self.normalize_reverse(array, mean, std) if array.ndim >= 3
                else array for array in array_list]

    @staticmethod
    def str_to_list(list_as_str):
        return list_as_str.replace('[', '').replace(',', '').replace(']', '').split()

    @staticmethod
    def normalize(array, mean, std):
        return (array - mean + 1e-8) / (std + 1e-8)

    @staticmethod
    def normalize_reverse(array, mean, std):
        return (array + mean - 1e-8) * (std - 1e-8)

    def _from_df(self, filename):
        stats = self.stats[self.stats['img_name'] == filename]
        if self.object:
            mean = self.str_to_list(stats['mean'].iat[0])
            mean = np.array([float(m) for m in mean])
            mean = mean[:, None, None]  # match dimensions to images

            std = self.str_to_list(stats['std'].iat[0])
            std = np.array([float(s) for s in std])
            std = std[:, None, None]
        else:
            mean = stats['mean'].iat[0]
            std = stats['std'].iat[0]
        return mean, std

    def _from_dict(self, filename):
        stats = {k: np.array(v).astype(np.float) for k, v in self.stats.items()}
        mean = stats['mean']
        mean = mean[:, None, None]
        std = stats['std']
        std = std[:, None, None]
        return mean, std

