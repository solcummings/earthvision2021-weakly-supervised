import numpy as np

from utils.transforms.transform_base import Transform


class VerticalFlip(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, array_list: list[np.ndarray], *args, **kwargs):
        return [self.flip(array) if array.ndim >= 2 else array for array in array_list]

    def backward(self, array_list: list[np.ndarray], *args, **kwargs):
        return [self.flip(array) if array.ndim >= 2 else array for array in array_list]

    @staticmethod
    def flip(array):
        return array.copy()[..., ::-1, :]

