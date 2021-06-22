import torch
import numpy as np

from utils.datasets import train_dataset
from utils.datasets import test_dataset


def build(dataset_name, dataset_args, **kwargs):
    dataset_dict = {
            'train': train_dataset.Dataset,
            'val': train_dataset.Dataset,
            'test': test_dataset.Dataset,  # predict
    }
    dataset = dataset_dict[dataset_name.lower()](**dataset_args)
    # worker_init_fn sets numpy seeds per worker because dataloader copies numpy seeds
    # to all child processes, batches will otherwise have identical random numpy values
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_args['batch_size'],
            shuffle=dataset_args['shuffle'],
            num_workers=dataset_args['num_workers'],
            pin_memory=True,
            worker_init_fn=lambda i: np.random.seed(torch.initial_seed() // 2 ** 32 + i),
    )
    return dataloader


