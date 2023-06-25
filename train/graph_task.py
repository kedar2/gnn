import torch
import torch_geometric
from tqdm import tqdm
from typing import Optional
from config import DataSplit, ExperimentConfig
from train.register import model_register, scheduler_register, optimizer_register, loss_fn_register
from math import inf

def train_loop(model: torch.nn.Module,
               loader: torch_geometric.loader.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: torch.device = "cpu"):
    """
    Performs a single training epoch for a graph-level task.
    """

    model.train()
    total_loss = 0
    for graph in tqdm(loader):
        graph = graph.to(device)
        optimizer.zero_grad()
        out = model(graph)
        loss = loss_fn(out, graph.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

def eval_loop(model: torch.nn.Module,
              metric: str,
              loader: torch_geometric.loader.DataLoader,
              device: torch.device = "cpu",
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """
    Performs a single evaluation epoch for a graph-level task.
    """

    model.eval()
    if metric == "acc":
        correct = 0
        total = 0
        for graph in tqdm(loader):
            graph = graph.to(device)
            with torch.no_grad():
                out = model(graph)
            pred = out.argmax(dim=1)
            correct += pred.eq(graph.y).sum().item()
            total += len(graph.y)
        metric_value = correct / total
    elif metric == "mse":
        mse = 0
        for graph in tqdm(loader):
            graph = graph.to(device)
            with torch.no_grad():
                out = model(graph)
            mse += torch.nn.functional.mse_loss(out, graph.y, reduction="sum").item()
        metric_value = mse / len(loader.dataset)
    else:
        raise NotImplementedError(f"Metric {metric} not implemented.")
    if scheduler is not None:
        scheduler.step(metric_value)
    return metric_value
    

def run_experiment(dataset: torch_geometric.data.Data,
                   split: DataSplit,
                   cfg: ExperimentConfig):
    """
    Runs a single experiment, training a model on a graph
    classification/regression task.

    Args:
        dataset: The dataset to use.
        split: The train/val/test split to use.
        cfg: The experiment configuration.
    """

    # Set up the model, optimizer, scheduler, loss function, and metric.
    model = model_register[cfg.model](cfg).to(cfg.device)
    optimizer = optimizer_register[cfg.optimizer](model.parameters(), lr=cfg.learning_rate)
    scheduler = scheduler_register[cfg.scheduler](optimizer, mode=cfg.goal, factor=0.5, verbose=True)
    loss_fn = loss_fn_register[cfg.loss_fn]()
    metric_name = cfg.metric
    if cfg.wandb:
        import wandb
        wandb.init(project="gnn", config=cfg.asdict())

    # Get the train/val/test loaders.
    train_loader = torch_geometric.loader.DataLoader(split.train_dataset, batch_size=cfg.batch_size, shuffle=True)
    validation_loader = torch_geometric.loader.DataLoader(split.validation_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = torch_geometric.loader.DataLoader(split.test_dataset, batch_size=cfg.batch_size, shuffle=False)


    if cfg.goal == "max":
        best_validation_metric = -inf
    elif cfg.goal == "min":
        best_validation_metric = inf
    else:
        raise NotImplementedError(f"Goal {cfg.goal} not recognized; must be 'max' or 'min'.")
    
    num_bad_epochs = 0
    # Run the training loop.
    for epoch in range(cfg.max_epochs):
        train_loop(model, train_loader, optimizer, loss_fn, cfg.device)
        train_metric = eval_loop(model, cfg.metric, train_loader, cfg.device)
        validation_metric = eval_loop(model, cfg.metric, validation_loader, cfg.device, scheduler)
        test_metric = eval_loop(model, cfg.metric, test_loader, cfg.device)

        if cfg.wandb:
            wandb.log({f"train_{metric_name}": train_metric,
                       f"validation_{metric_name}": validation_metric,
                       f"test_{metric_name}": test_metric})
        if cfg.goal == "max" and validation_metric > best_validation_metric:
            best_validation_metric = validation_metric
            best_validation_str = " (new best validation)"
            num_bad_epochs = 0
        elif cfg.goal == "min" and validation_metric < best_validation_metric:
            best_validation_metric = validation_metric
            best_validation_str = " (new best validation)"
            num_bad_epochs = 0
        else:
            best_validation_str = ""
            num_bad_epochs += 1
        print(f"Epoch {epoch} | Train {metric_name}: {train_metric:.4f} | Validation {metric_name}: {validation_metric:.4f}{best_validation_str} | Test {metric_name}: {test_metric:.4f}")
        if num_bad_epochs >= cfg.stopping_patience:
            print(f"Stopping early after {epoch} epochs.")
            break


