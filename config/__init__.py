from dataclasses import dataclass, asdict
from typing import Optional
from torch import Tensor
from torch_geometric.data import Data

@dataclass
class ExperimentConfig:
    dataset: str = "Cora" # Name of the dataset to be used.
    task: Optional[str] = None # Task to be performed on the dataset (e.g. node_classification).
    model: str = "GNN" # Name of model to be used.
    input_dim: Optional[int] = None # Dimension of input features.
    output_dim: Optional[int] = None # Dimension of output features.
    hidden_dim: int = 64 # Dimension of hidden layers.
    num_hidden_layers: int = 1 # Number of hidden layers.
    residual_connections: bool = True # Whether to use residual connections in the GNN.
    last_layer_fa: bool = False # Whether to make the last layer fully connected.
    layer_type: str = "GCN" # Type of convolution layer to be used.
    act_fn: str = "ReLU" # Activation function to be used.
    dropout: float = 0.0 # Dropout rate.
    loss_fn: str = "CrossEntropyLoss" # Loss function to be used for the model.
    device: Optional[str] = None # Device to be used for training.
    optimizer: str = "Adam" # Optimizer to be used for training.
    scheduler: str = "ReduceLROnPlateau" # Scheduler to be used for training.
    learning_rate: float = 0.01 # Learning rate for the optimizer.
    batch_size: int = 64 # Batch size for training.
    max_epochs: int = 400 # Maximum number of epochs to train for.
    stopping_patience: int = 100 # Number of epochs to wait without improvement before early stopping.
    metric: Optional[str] = None # Metric to be used for early stopping (e.g. acc)
    goal: Optional[str] = None # Goal of the metric (max or min).
    wandb: bool = False # Whether to use wandb for logging.
    
    def asdict(self):
        return asdict(self)

@dataclass
class DataSplit:
    train_mask: Optional[Tensor] = None
    validation_mask: Optional[Tensor] = None
    test_mask: Optional[Tensor] = None
    train_dataset: Optional[Data] = None
    validation_dataset: Optional[Data] = None
    test_dataset: Optional[Data] = None