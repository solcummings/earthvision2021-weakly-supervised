import numpy as np
import torch

from utils.transforms.transform_base import Transform


class ToTensor(Transform):
    # probability should be binary
    def __init__(self, p=1., *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    def forward(self, array_list: list[np.ndarray], *args, **kwargs) \
            -> list[torch.Tensor]:
        # wrap array in array again to avoid typeerror
        return [torch.from_numpy(np.array(array)) for array in array_list]

    def backward(self, array_list: list[torch.Tensor], *args, **kwargs) \
            -> list[np.ndarray]:
        return [array.numpy() for array in array_list]

