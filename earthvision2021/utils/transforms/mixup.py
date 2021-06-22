import numpy as np

from utils.transforms.transform_base import Transform


class Mixup(Transform):
    def __init__(self, mix_labels=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mix_labels = mix_labels

    def forward(self, array_list: list[np.ndarray], mix_array_list: list[np.ndarray],
            *args, **kwargs):
        return [self.mix(array, mix_array) if self.mix_labels else array for
                array, mix_array in zip(array_list, mix_array_list)]

    backward = forward

    # assumes arrays are one-hot
    @staticmethod
    def mix(array, mix_array, alpha=1.):
        _lambda = np.random.beta(alpha, alpha) if alpha > 0 else 1
        output_array = _lambda * array + (1 - _lambda) * mix_array
        return output_array

