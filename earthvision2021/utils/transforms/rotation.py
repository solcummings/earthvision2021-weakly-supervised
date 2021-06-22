import numpy as np

from utils.transforms.transform_base import Transform


class Rotation(Transform):
    def __init__(self, k=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def forward(self, array_list: list[np.ndarray], *args, **kwargs):
        return [self.rotate(array, self.k) if array.ndim >= 2 else array for array in
                array_list]

    def backward(self, array_list: list[np.ndarray], *args, **kwargs):
        return [self.rotate(array, 4 - self.k) if array.ndim >= 2 else array for array in
                array_list]

    @staticmethod
    def rotate(array, k):
        return np.rot90(array.copy(), k=k, axes=(-1, -2))

