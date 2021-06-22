import numpy as np

from utils.transforms.normalize import Normalize
from utils.transforms.to_tensor import ToTensor
from utils.transforms.pad import Pad
from utils.transforms.center_crop import CenterCrop
from utils.transforms.random_crop import RandomCrop
from utils.transforms.horizontal_flip import HorizontalFlip
from utils.transforms.vertical_flip import VerticalFlip
from utils.transforms.rotation import Rotation
from utils.transforms.cutmix import Cutmix
from utils.transforms.mixup import Mixup


class ComposeTransforms:
    # automatically normalizes and converts to tensor
    def __init__(self, transforms_dict, mean_std, normalize=True, to_tensor=True,
            debug=False):
        implemented_dict = {
            'pad': Pad,
            'centercrop': CenterCrop,
            'randomcrop': RandomCrop,
            'hflip': HorizontalFlip,
            'vflip': VerticalFlip,
            'rotation': Rotation,
            'cutmix': Cutmix,
            'mixup': Mixup,
        }
        mix_list = ['cutmix', 'mixup']
        self.transforms_list = [implemented_dict[k](**v) for k, v in
                transforms_dict.items()]
        self.mix = True if any([t in mix_list for t in transforms_dict.keys()]) else False
        self.transforms_list_reverse = self.transforms_list[::-1]

        self.normalize = Normalize(mean_std, disable=not normalize)
        self.to_tensor = ToTensor(disable=not to_tensor)
        if debug:
            self.__debug()

    def __call__(self, x: list[np.ndarray], name, mix_x: list[np.ndarray] = None,
            reverse=False):
        if reverse:
            # reverse skips normalization and totensor
            for t in self.transforms_list_reverse:
                x = t(x, reverse=True)
        else:
            for t in self.transforms_list:
                x = t(x, mix_array_list=mix_x)
            x = self.normalize(x, filename=name)
            x = self.to_tensor(x)
        return x

    def __repr__(self):
        transforms_list = [str(t) for t in self.transforms]
        return str(transforms_list)

    def __debug(self):
        channel1 = np.arange(4)
        channel2 = channel1.copy()
        channel2[0] = 1
        channel3 = channel1.copy()
        channel3[0] = 2
        # img is c, h, w & label is h, w
        img = np.stack([channel1, channel2, channel3]).reshape(3, 2, 2)
        label = img[0].copy() * 10

        mix_img = img.copy() * 2
        mix_label = mix_img[0].copy() * 10

        array_list = [img, label]
        mix_array_list = [mix_img, mix_label]
        import pdb;pdb.set_trace()
        for t in self.transforms_list:
            array_list = t(array_list, mix_array_list)
            print(array_list)
        exit()

