import torch


def build(optimizer, scheduler_name, scheduler_args, **kwargs):
    for k, v in scheduler_args.items():
        if k in ['eta_min']:
            scheduler_args[k] = float(v)

    scheduler_dict = {
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
            'cosine_wr': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }
    return scheduler_dict[scheduler_name.lower()](optimizer, **(scheduler_args or {}))

