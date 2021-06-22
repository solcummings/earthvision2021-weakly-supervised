import numpy as np

from utils.transforms.transform_base import Transform


class Pad(Transform):
    def __init__(self, patchsize, p=1., *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        self.patchsize = patchsize

    def forward(self, array_list: list[np.ndarray], *args, **kwargs):
        output_list = []
        for array in array_list:
            if array.ndim >= 2:
                array = self.zero_pad(array, self.patchsize) if \
                        self.determine_to_pad(array, self.patchsize) else array
            output_list.append(array)
        return output_list

    backward = forward

    @staticmethod
    def determine_to_pad(array, patchsize):
        h, w = array.shape[-2], array.shape[-1]
        return True if patchsize > h or patchsize > w else False

    @staticmethod
    def zero_pad(array, patchsize, mode='constant'):
        h, w = array.shape[-2], array.shape[-1]
        pad_h = max(int(np.ceil((patchsize - h) / 2)), 0)
        pad_w = max(int(np.ceil((patchsize - w) / 2)), 0)
        # pad for arrays with channels
        pad_tuple = ((0, 0),) * (array.ndim - 2) + ((pad_h, pad_h), (pad_w, pad_w))
        array = np.pad(array.copy(), pad_tuple, mode=mode)
        array = array[..., :patchsize, :patchsize]
        return array

