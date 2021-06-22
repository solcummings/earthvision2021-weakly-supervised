import random
import torch
import numpy as np


def set_seeds(seed, deterministic=False):
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # deprecated
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False if deterministic else True

    np.random.seed(seed)

    random.seed(seed)

