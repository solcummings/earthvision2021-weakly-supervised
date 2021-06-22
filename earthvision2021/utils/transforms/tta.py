import numpy as np
import torch

from itertools import combinations

from utils.transforms.compose import ComposeTransforms


class TTA:
    # assumes normalization and crop are done
    def __init__(self, transforms_dict, debug=False):
        self.transforms_pattern = self.compose_tta_patterns(transforms_dict)
        # include original with no augmentations
        self.transforms_pattern = [
                ComposeTransforms({}, mean_std=None, normalize=False, to_tensor=True),
        ] + self.transforms_pattern
        if debug:
            self.__debug()

    # all transforms take inputs as list of numpy arrays
    def __call__(self, x: torch.tensor, *args, **kwargs):
        # change type to list[np.ndarray] for transforms, default collate_fn automatically
        # converts numpy arrays to torch tensors, so each must be reverted for transforms
        x = [i.numpy() for i in x] if isinstance(x, list) else [x.numpy()]
        augmented_input_list = []
        for t in self.transforms_pattern:
            augmented_input_list.append(t(x, name=None))
        return augmented_input_list

    def compose_tta_patterns(self, pattern_dict: dict) -> list:
        all_patterns_list = []
        for i in range(len(pattern_dict.keys())):
            patterns = [list(p) for p in combinations(pattern_dict.keys(), i + 1)]
            for p in patterns:
                transform_dict = {k: pattern_dict[k] for k in p}
                all_patterns_list.append(ComposeTransforms(transform_dict, mean_std=None,
                    normalize=False, to_tensor=True))
        return all_patterns_list

    def reverse(self, x: list[np.ndarray]):
        reversed_inputs = []
        for t, augmented_input in zip(self.transforms_pattern, x):
            reversed_inputs.append(t(augmented_input, name=None, reverse=True))
        return reversed_inputs

    def __repr__(self):
        transforms_pattern = [str(t) for t in self.transforms_pattern]
        return str(transforms_pattern)

    def __debug(self):
        channel1 = np.arange(4)
        channel2 = channel1.copy()
        channel2[0] = 1
        channel3 = channel1.copy()
        channel3[0] = 2
        # img is c, h, w & label is h, w
        img = np.stack([channel1, channel2, channel3]).reshape(3, 2, 2)
        label = img[0].copy() * 10

        array_list = [img, label]
        import pdb;pdb.set_trace()
        output_list = []
        for t in self.transforms_pattern:
            output_array = t(array_list, name=None, reverse=True)
            print(output_array)
            output_list.append(output_array)
        exit()

