import argparse
import torch
from typing import NamedTuple, Optional
from dataclasses import dataclass, asdict
import json

@dataclass
class Configuration:
    """
    Class storing settings for models and training.
    """
    config_file: str = None # Path to YAML configuration file.
    learning_rate: float = 0.001 # Learning rate for the optimizer.
    max_epochs: int = 100 # Maximum number of epochs before stopping.
    layer_type: str = 'GCN' # Type of layer to use for the model.
    display: bool = True # Whether or not to print progress.
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' # Name of device to use for training.
    eval_every: int = 1 # Number of epochs between evaluations.
    dropout: float = 0.0 # Dropout rate for the model.
    weight_decay: float = 0.0005 # Weight decay for the optimizer.
    input_dim: Optional[int] = None # Dimension of the input features.
    hidden_dim: int = 64 # Dimension of the hidden layers of the model.
    output_dim: Optional[int] = None # Dimension of the output features.
    num_hidden_layers: int = 2 # Number of hidden layers in the model.
    batch_size: int = 32 # Batch size for the model.
    dataset: str = None # Name of the dataset to use for the model.
    loss_fn: str = None # Loss function to use for training.
    last_layer_fa: bool = False # Whether or not to make the last layer fully adjacent.
    metric: str = None # Metric to use for evaluation.
    goal: str = "max" # Goal of the metric ("max" to maximize and "min" to minimize).
    task: str = None # Type of task of the dataset.
    wandb: bool = False # Whether or not to use wandb for logging.
    sweep: bool = False # Whether or not to perform a hyperparameter sweep.
    additional_metrics: list = None # Additional metrics to use for evaluation.

    def asdict(self):
        return asdict(self)
    def update(self, args: dict):
        for key, value in args.items():
            if not hasattr(self, key):
                raise ValueError("Invalid setting: {}".format(key))
            setattr(self, key, value)
    def set_defaults(self):
        if not self.dataset:
            return
        settings = json.load(open("config/defaults.json", 'r'))
        if self.dataset in settings:
            dataset_settings = settings[self.dataset]
            for key, value in dataset_settings.items():
                if not hasattr(self, key):
                    raise ValueError("Invalid setting: {}".format(key))
                if not getattr(self, key):
                    setattr(self, key, value)

def parse_bool(s: str) -> bool:
    """
    Parses a string into a boolean value.

    Args:
        s (str): String to parse.
    
    Returns:
        bool: Boolean value of the string.
    """
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError("Invalid value for boolean argument: {}".format(s))

def parse_settings() -> dict:
    """
    Parses command line arguments and returns them as a dict.

    Returns:
        args (dict): Dictionary of settings.
    """
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--layer_type', type=str)
    parser.add_argument('--display', type=parse_bool)
    parser.add_argument('--device', type=str)
    parser.add_argument('--eval_every', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--num_hidden_layers', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--loss_fn', type=str)
    parser.add_argument('--last_layer_fa', type=parse_bool)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--goal', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--wandb', type=parse_bool)

    args = vars(parser.parse_args())

    return args