from fosr.preprocessing import fosr, sdrf
import torch
from torch_geometric.data import InMemoryDataset, download_url

class RewiredDataset(InMemoryDataset):
    pass

def get_rewired_dataset(dataset, rewiring, filename=None):
    """
    Produces a rewired PyG dataset, and stores the rewiring information in a file.
    If the filename is already taken, attempts to load the dataset from there.
    """
    L = list(dataset)
    L[0].x = 5
    print(L)
    raise NotImplementedError
