import torch_geometric
from torch_geometric.transforms import Constant
from torch.utils.data import random_split
from config import ExperimentConfig, DataSplit
from typing import Tuple

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

def load_experiment(cfg_dict: dict = None) -> Tuple[torch_geometric.data.Dataset, ExperimentConfig, DataSplit]:
    """
    Loads a dataset and creates a train/validation/test split.

    Args:
        cfg_dict (dict): Dictionary containing the settings for the experiment.
    
    Returns:
        dataset (torch_geometric.data.Dataset): The dataset.
        cfg (ExperimentConfig): The experiment configuration.
        split (DataSplit): The train/validation/test split. 
    """

    cfg = ExperimentConfig(**cfg_dict)
    dataset_name = cfg.dataset
    if dataset_name in ["REDDIT-BINARY", "IMDB-BINARY", "COLLAB", "ENZYMES", "PROTEINS", "MUTAG"]:
        from torch_geometric.datasets import TUDataset

        # add constant node features for datasets without node features
        if dataset_name in ["REDDIT-BINARY", "IMDB-BINARY", "COLLAB"]:
            dataset = TUDataset(root='data', name=dataset_name, transform=Constant())
        else:
            dataset = TUDataset(root='data', name=dataset_name)
        
        train_dataset, validation_dataset, test_dataset = train_val_test_split(dataset)
    elif dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root='data', name=dataset_name)
        graph = dataset.data
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    
    if cfg.metric is None:
        cfg.metric = "acc"
        cfg.goal = "max"
    if cfg.input_dim is None:
        cfg.input_dim = dataset.num_features
    if cfg.output_dim is None:
        cfg.output_dim = dataset.num_classes
    if "graph" in cfg.task:
        split = DataSplit(train_dataset=train_dataset,
                            validation_dataset=validation_dataset,
                            test_dataset=test_dataset)
    elif "node" in cfg.task:
        split = DataSplit(train_mask=graph.train_mask,
                            validation_mask=graph.val_mask,
                            test_mask=graph.test_mask)
    else:
        raise NotImplementedError(f"Task {cfg.task} not supported.")
        

    return dataset, cfg, split