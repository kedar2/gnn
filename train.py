import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from math import inf
from tqdm import tqdm
from config.parsing import Configuration
from models.gnn import GNN
import wandb

class Experiment:
    """
    Experiment class for executing training and testing loops.

    Args:
        dataset (torch_geometric.data.Dataset): Full dataset.
        train_dataset (torch_geometric.data.Dataset): Dataset to be used for training.
        validation_dataset (torch_geometric.data.Dataset): Dataset to be used for validation.
        test_dataset (torch_geometric.data.Dataset): Dataset to be used for testing.
        train_mask (torch.Tensor): Mask for selecting training data (e.g. for node classification)
        validation_mask (torch.Tensor): Mask for selecting validation data.
        test_mask (torch.Tensor): Mask for selecting testing data.
        cfg (Configuration): Configuration object containing experiment settings.
    """
    def __init__(self,
                train_dataset: torch_geometric.data.Dataset=None,
                validation_dataset: torch_geometric.data.Dataset=None,
                test_dataset: torch_geometric.data.Dataset=None,
                train_mask: torch.Tensor=None,
                validation_mask: torch.Tensor=None,
                test_mask: torch.Tensor=None,
                cfg: Configuration=None):
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.train_mask = train_mask
        self.validation_mask = validation_mask
        self.test_mask = test_mask
        self.max_epochs = cfg.max_epochs
        self.metric = cfg.metric
        self.goal = cfg.goal
        self.device = cfg.device
        self.display = cfg.display
        self.eval_every = cfg.eval_every
        self.task = cfg.task
        self.wandb = cfg.wandb
        self.sweep = cfg.sweep
        self.additional_metrics = cfg.additional_metrics

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        if cfg.input_dim is None:
            cfg.input_dim = self.train_dataset.num_features
        if cfg.output_dim is None:
            cfg.output_dim = self.train_dataset.num_classes

        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TODO: Pass arguments directly to GNN instead of using cfg.
        self.model = GNN(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if cfg.loss_fn == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss function {cfg.loss_fn} not implemented.")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001, verbose=True)

    def run(self):
        if self.wandb and not self.sweep:
            wandb.init(project="gnn", config=self.cfg.asdict())

        # Keep track of progress to determine when to stop.
        if self.goal == 'max':
            best_validation_metric = -inf
        elif self.goal == 'min':
            best_validation_metric = inf
        else:
            raise ValueError(f"Goal {self.goal} is not supported (must be either 'max' or 'min').")


        for epoch in range(1, 1 + self.max_epochs):
            # Train for one epoch.
            if self.display:
                print(f"====Epoch {epoch}\nTraining...")
            self.train()
            
            new_best_str = ''
            if epoch % self.eval_every == 0:
                # Evaluate, report progress, and check whether to stop training.
                if self.display:
                    print(f"\nEvaluating...")
                train_metric, validation_metric, test_metric = self.eval()

                if (validation_metric > best_validation_metric and self.goal == 'max') or (validation_metric < best_validation_metric and self.goal == 'min'):
                    # Checks if the validation metric is the best so far.
                    # best_train_metric and best_test_metric are defined as the train and test loss when the validation metric is the best.
                    best_train_metric = train_metric
                    best_validation_metric = validation_metric
                    best_test_metric = test_metric
                    new_best_str = ' (new best validation {self.metric})'
                if self.display:
                    print(f'\nTrain {self.metric}: {train_metric}, Validation {self.metric}: {validation_metric}{new_best_str}, Test: {test_metric}\n')
        if self.display:
            print('\nReached max epoch count, stopping training')
            print(f'Best train {self.metric}: {best_train_metric}, Best validation {self.metric}: {best_validation_metric}, Best test {self.metric}: {best_test_metric}')
        
        if self.wandb:
            wandb.log({f"best_train_{self.metric}": best_train_metric, f"best_validation_{self.metric}": best_validation_metric, f"best_test_{self.metric}": best_test_metric})
        return train_metric, validation_metric, test_metric
    def train(self):
        # Train loop    
        self.model.train()
        total_loss = 0
        if self.task in ["graph_classification", "graph_regression"]:
            for graph in tqdm(self.train_loader, disable=(not self.display)):
                self.optimizer.zero_grad()
                graph = graph.to(self.device)
                pred = self.model(graph)
                loss = self.loss_fn(input=pred, target=graph.y)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
        else:
            raise NotImplementedError("Task {self.task} not implemented.}")
        self.scheduler.step(total_loss)
    def eval(self) -> float:
        # Evaluation loop
        self.model.eval()
        loaders = [self.train_loader, self.validation_loader, self.test_loader]
        metrics = []
        epoch_metrics = {}
        for split_name, loader in zip(["train", "validation", "test"], loaders):
            if self.metric == "accuracy":
                sample_size = len(loader.dataset)
                total_correct = 0
                with torch.no_grad():
                    for graph in tqdm(loader, disable=(not self.display)):
                        graph = graph.to(self.device)
                        out = self.model(graph)
                        _, pred = out.max(dim=1)
                        total_correct += pred.eq(graph.y).sum().item()
                acc = total_correct / sample_size
                epoch_metrics[f"{split_name}_accuracy"] = acc
                metrics.append(acc)
            else:
                raise NotImplementedError(f"Metric {self.metric} not implemented.")
        if self.wandb:
            wandb.log(epoch_metrics)
        return metrics

