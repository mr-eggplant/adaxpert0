import torch.utils.data as data


class CachedSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.cache = [self.dataset[i] for i in self.indices]

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.indices)
