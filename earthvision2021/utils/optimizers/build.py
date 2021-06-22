import torch


def build(params, optimizer_name, optimizer_args, **kwargs):
    for k, v in optimizer_args.items():
        if k in ['lr', 'weight_decay']:
            optimizer_args[k] = float(v)

    optimizer_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
    }
    return optimizer_dict[optimizer_name.lower()](params, **(optimizer_args or {}))

