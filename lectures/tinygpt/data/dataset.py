import torch
from torch.utils.data import Dataset


def split_at_last(lst, element):
    # Find the last occurrence of the element in the list
    idx = len(lst) - 1 - lst[::-1].index(element) if element in lst else -1

    # If the element is found in the list, split it. Otherwise, return the original list and an empty list.
    if idx != -1:
        return lst[:idx], lst[idx:]
    else:
        return lst, []


class CharDataset(Dataset):
    def __init__(self, config, data):
        self.config = config
        self.data = data  # alread batched

    def __len__(self):
        return len(self.data) - self.config.n_block

    def __getitem__(self, idx):

        # grab a chunk of (n_block + 1) characters from the data
        chunk = self.data[idx][0:self.config.n_block+1]
        if (len(chunk) == 0):
            print(f'Got empy chunk at {idx} before split')
            print(self.data[idx])
        chunk, _ = split_at_last(chunk, 13)

        if (len(chunk) == 0):
            print(f'Got empy chunk at {idx} after split')

        # pad the input to the block size
        while len(chunk) < self.config.n_block+1:
            chunk.append(self.config.padding_idx)

        # return as tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
