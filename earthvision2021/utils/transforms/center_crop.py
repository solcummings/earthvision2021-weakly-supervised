import numpy as np

from utils.transforms.transform_base import Transform
from utils.transforms.pad import Pad


class CenterCrop(Transform):
    # probability should be binary
    def __init__(self, patchsize, p=1., *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        self.patchsize = patchsize

    def forward(self, array_list: list[np.ndarray], *args, **kwargs):
        output_list = []
        for array in array_list:
            if array.ndim >= 2:
                array = Pad.zero_pad(array, self.patchsize) if \
                        Pad.determine_to_pad(array, self.patchsize) else \
                        self.crop(array, self.patchsize, self.patchsize)
            output_list.append(array)
        return output_list

    backward = forward

    @staticmethod
    def crop(array, h_cropsize, w_cropsize):
        h, w = array.shape[-2], array.shape[-1]
        h_start = (h - h_cropsize) // 2
        w_start = (w - w_cropsize) // 2
        return array.copy()[..., h_start:h_start+h_cropsize, w_start:w_start+w_cropsize]

