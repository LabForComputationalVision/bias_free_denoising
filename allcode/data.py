import os
import os.path
import numpy as np
import h5py
import torch


DATASET_REGISTRY = {}


def build_dataset(name, *args, **kwargs):
    return DATASET_REGISTRY[name](*args, **kwargs)


def register_dataset(name):
    def register_dataset_fn(fn):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = fn
        return fn

    return register_dataset_fn


@register_dataset("bsd400")
def load_bsd400(data, batch_size=100, num_workers=0):
    data = os.path.join(data, 'bsd400')
    train_dataset = Dataset(filename=os.path.join(data, "train.h5"))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

    valid_dataset = Dataset(filename=os.path.join(data, "valid.h5"))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False)
    return train_loader, valid_loader, None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.keys = list(self.h5f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)
