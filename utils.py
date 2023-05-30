import torch_geometric
from torch.utils.data import random_split

def train_val_test_split(dataset: torch_geometric.data.Dataset,
                         train_ratio: float=0.8,
                         validation_ratio: float=0.1,
                         test_ratio: float=0.1):
    """
    Split a dataset into train, validation, and test sets.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset to be split.
        train_ratio (float): Ratio of examples to be used for training.
        validation_ratio (float): Ratio of examples to be used for validation.
        test_ratio (float): Ratio of examples to be used for testing.
    """
    assert train_ratio + validation_ratio + test_ratio == 1.0
    num_examples = len(dataset)
    num_train = int(train_ratio * num_examples)
    num_validation = int(validation_ratio * num_examples)
    num_test = num_examples - num_train - num_validation
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [num_train, num_validation, num_test])
    return train_dataset, validation_dataset, test_dataset