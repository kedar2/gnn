import argparse
import torch
from typing import NamedTuple

class Configuration(NamedTuple):
    """
    Configuration settings for the experiment.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        max_epochs (int): Maximum number of epochs before stopping.
        layer_type (str): Type of layer to use for the model.
        display (bool): Whether or not to print progress.
        device (torch.device): Name of device to use for training.
        eval_every (int): Number of epochs between evaluations.
        patience (int): Number of epochs to wait before early stopping.
        train_fraction (float): Fraction of the dataset to be used for training.
        validation_fraction (float): Fraction of the dataset to be used for validation.
        test_fraction (float): Fraction of the dataset to be used for testing.
        dropout (float): Dropout rate for the model.
        weight_decay (float): Weight decay for the optimizer.
        input_dim (int): Dimension of the input layer of the model.
        hidden_dim (int): Dimension of the hidden layers of the model.
        output_dim (int): Dimension of the output layer of the model.
        num_hidden_layers (int): Number of hidden layers in the model.
        batch_size (int): Batch size for the model.
        num_trials (int): Number of trials to run for the model.
        rewiring (str): Type of rewiring to use for the model.
        num_iterations (int): Number of iterations to run for the model.
        alpha (float): alpha hyperparameter for DIGL.
        k (int): k hyperparameter for DIGL.
        eps (float): eps hyperparameter for DIGL.
        dataset (str): Name of the dataset to use for the model.
        benchmark (str): Name of the benchmark to use for the model.
        last_layer_fa (bool): Whether or not to make the last layer fully adjacent.
        lower_criterion_is_better (bool): Whether or not a lower criterion is better (e.g. if criterion is accuracy, then higher is better)
        num_relations (int): Number of relations to use for R-GNNs.
        task_type (str): Type of task to run (e.g. 'graph_classification', 'node_regression').
        criterion_name (str): Name of the criterion used for validation.
        wandb (bool): Whether or not to use Weights and Biases to log results.
    """
    learning_rate: float = 0.001
    max_epochs: int = 1000
    layer_type: str = 'GCN'
    display: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_every: int = 1
    patience: int = 10
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    dropout: float = 0.5
    weight_decay: float = 0.0005
    input_dim: int = 1
    hidden_dim: int = 64
    output_dim: int = 1
    num_hidden_layers: int = 2
    batch_size: int = 32
    num_trials: int = 10
    rewiring: str = 'FoSR'
    num_iterations: int = 10
    alpha: float = 0.0015
    k: int = 10
    eps: float = 0.1
    dataset: str = ''
    benchmark: str = ''
    last_layer_fa: bool = False
    lower_criterion_is_better: bool = True
    num_relations: int = 1
    task_type: str = 'graph_classification'
    criterion_name: str = 'accuracy'
    wandb: bool = True

def parse_bool(s: str) -> bool:
    """
    Parses a string into a boolean value.
    """
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError("Invalid value for boolean argument: {}".format(s))

def get_args_from_input() -> dict:
    """
    Parses command line arguments and returns them as a dict.
    """
    parser = argparse.ArgumentParser(description='modify network parameters', argument_default=argparse.SUPPRESS)
    parser.add_argument('--learning_rate', metavar='', type=float, help='learning rate')
    parser.add_argument('--max_epochs', metavar='', type=int, help='maximum number of epochs for training')
    parser.add_argument('--layer_type', metavar='', help='type of layer in GNN (GCN, GIN, GAT, etc.)')
    parser.add_argument('--display', metavar='', type=parse_bool, help='toggle display messages showing training progress')
    parser.add_argument('--device', metavar='', type=str, help='name of CUDA device to use or CPU')
    parser.add_argument('--eval_every', metavar='X', type=int, help='calculate validation/test accuracy every X epochs')
    parser.add_argument('--patience', metavar='P', type=int, help='model stops training after P epochs with no improvement')
    parser.add_argument('--train_fraction', metavar='', type=float, help='fraction of the dataset to be used for training')
    parser.add_argument('--validation_fraction', metavar='', type=float, help='fraction of the dataset to be used for validation')
    parser.add_argument('--test_fraction', metavar='', type=float, help='fraction of the dataset to be used for testing')
    parser.add_argument('--dropout', metavar='', type=float, help='layer dropout probability')
    parser.add_argument('--weight_decay', metavar='', type=float, help='weight decay added to loss function')
    parser.add_argument('--hidden_dim', metavar='', type=int, help='width of hidden layer')
    parser.add_argument('--num_hidden_layers', metavar='', type=int, help='number of hidden layers')
    parser.add_argument('--batch_size', metavar='', type=int, help='number of samples in each training batch')
    parser.add_argument('--num_trials', metavar='', type=int, help='number of times the network is trained'),
    parser.add_argument('--rewiring', metavar='', type=str, help='type of rewiring to be performed'),
    parser.add_argument('--num_iterations', metavar='', type=int, help='number of iterations of rewiring')
    parser.add_argument('--alpha', type=float, help='alpha hyperparameter for DIGL')
    parser.add_argument('--k', type=int, help='k hyperparameter for DIGL')
    parser.add_argument('--eps', type=float, help='epsilon hyperparameter for DIGL')
    parser.add_argument('--dataset', type=str, help='name of dataset to use')
    parser.add_argument('--benchmark', type=str, help='name of benchmark to run (eg. tudataset, ogb)')
    parser.add_argument('--last_layer_fa', type=parse_bool, help='whether or not to make last layer fully adjacent')
    parser.add_argument('--wandb', type=parse_bool, help='whether or not to use wandb to log results')
    arg_values = vars(parser.parse_args())
    return arg_values