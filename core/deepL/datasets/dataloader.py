import random

from torch import initial_seed
from torch.utils.data import DataLoader
from numpy import random as np_random



def reset_seed_worker_init_fn(worker_id):
    """Reset NumPy and Python seed for data loader worker."""
    seed = initial_seed() % (2 ** 32)
    np_random.seed(seed)
    random.seed(seed)

def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=None,
    collate_fn=None,
    sampler=None,
    pin_memory=False,
    drop_last=False,
):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader
