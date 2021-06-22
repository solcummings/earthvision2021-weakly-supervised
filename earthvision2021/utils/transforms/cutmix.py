import numpy as np

from utils.transforms.transform_base import Transform
from utils.transforms.random_crop import RandomCrop
from utils.transforms.mixup import Mixup


class Cutmix(Transform):
    # preserves center if centersize > 0
    # centersize is always square
    def __init__(self, cutsize_min=20, cutsize_max=32, centersize=0, mix_labels=True,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert cutsize_max >= cutsize_min
        self.cutsize = (cutsize_min, cutsize_max)
        self.centersize = centersize
        self.mix_labels = mix_labels

    def forward(self, array_list: list[np.ndarray], mix_array_list: list[np.ndarray],
            *args, **kwargs):
        h_cutsize = np.random.randint(self.cutsize[0], self.cutsize[1] + 1)
        w_cutsize = np.random.randint(self.cutsize[0], self.cutsize[1] + 1)
        h_coef = np.random.rand()
        w_coef = np.random.rand()
        output_list = []
        for array, mix_array in zip(array_list, mix_array_list):
            if array.ndim >= 2:
                array = self.cut(array, mix_array, h_cutsize, w_cutsize, h_coef, w_coef,
                        self.centersize)
            elif self.mix_labels:
                array = Mixup.mix(array, mix_array)
            else:
                # when labels are not augmented
                pass
            output_list.append(array)
        return output_list

    backward = forward

    @staticmethod
    def cut(array, mix_array, h_cutsize, w_cutsize, h_coef, w_coef, centersize):
        h, w = array.shape[-2], array.shape[-1]
        output_array = array.copy()
        if centersize > 0:
            center_h_start = int(np.floor(h / 2) - np.floor(centersize / 2))
            center_w_start = int(np.floor(w / 2) - np.floor(centersize / 2))
            center_array = array.copy()[
                        ...,
                        center_h_start:center_h_start+centersize,
                        center_w_start:center_w_start+centersize,
            ]
        mix_array = RandomCrop.crop(mix_array, h_cutsize, w_cutsize, h_coef, w_coef)

        h_start = int(np.floor((h - h_cutsize) * h_coef))
        w_start = int(np.floor((w - w_cutsize) * w_coef))
        output_array[
                ...,
                h_start:h_start+h_cutsize,
                w_start:w_start+w_cutsize,
        ] = mix_array
        if centersize > 0:
            output_array[
                    ...,
                    center_h_start:center_h_start+centersize,
                    center_w_start:center_w_start+centersize,
            ] = center_array
        return output_array

