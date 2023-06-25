import torch
import torch_geometric
from typing import Optional
from config import ExperimentConfig, DataSplit
from train.register import model_register, scheduler_register, optimizer_register, loss_fn_register
from math import inf

def train_loop(model: torch.nn.Module,
               graph: torch_geometric.data.Data,
               train_mask: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module):
    """
    Performs a single training epoch for a node-level task.
    """

    model.train()
    total_loss = 0
    optimizer.zero_grad()
    out = model(graph)[train_mask]
    loss = loss_fn(out, graph.y[train_mask])
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

def eval_loop(model: torch.nn.Module,
              metric: str,
              graph: torch_geometric.data.Data,
              eval_mask: torch.Tensor,
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """
    Performs a single evaluation epoch for a node-level task.
    """

    model.eval()
    if metric == "acc":
        with torch.no_grad():
            logits = model(graph)
        pred = logits.argmax(dim=1)
        correct = pred[eval_mask].eq(graph.y[eval_mask]).sum().item()
        metric_value = correct / eval_mask.sum().item()
    elif metric == "mse":
        with torch.no_grad():
            logits = model(graph)
        mse = torch.nn.functional.mse_loss(logits[eval_mask], graph.y[eval_mask], reduction="sum").item()
        metric_value = mse / eval_mask.sum().item()
    else:
        raise NotImplementedError(f"Metric {metric} not implemented.")
    if scheduler is not None:
        scheduler.step(metric_value)
    return metric_value
    

def run_experiment(graph: torch_geometric.data.Data,
                   split: DataSplit,                   
                   cfg: ExperimentConfig):
    """
    Runs a single experiment, training a model on a node
    classification/regression task.

    Args:
        graph (torch_geometric.data.Data): The graph to train on.
        split (DataSplit): The train/validation/test split.
        cfg (ExperimentConfig): The experiment configuration.
    """

    model = model_register[cfg.model](cfg).to(cfg.device)
    optimizer = optimizer_register[cfg.optimizer](model.parameters(), lr=cfg.learning_rate)
    loss_fn = loss_fn_register[cfg.loss_fn]()
    scheduler = scheduler_register[cfg.scheduler](optimizer,
                                                  mode=cfg.goal,
                                                  factor=0.5,
                                                  verbose=True)
    metric_name = cfg.metric
    if cfg.wandb:
        import wandb
        wandb.init(project="gnn", config=cfg.asdict())
    if cfg.goal == "max":
        best_validation_metric = -inf
    elif cfg.goal == "min":
        best_validation_metric = inf
    else:
        raise NotImplementedError(f"Goal {cfg.goal} not recognized; must be either 'max' or 'min'.")
    
    num_bad_epochs = 0
    for epoch in range(cfg.max_epochs):
        train_loop(model, graph, split.train_mask, optimizer, loss_fn)

        train_metric = eval_loop(model, cfg.metric, graph, split.train_mask)
        validation_metric = eval_loop(model, cfg.metric, graph, split.validation_mask, scheduler)
        test_metric = eval_loop(model, cfg.metric, graph, split.test_mask)
        
    
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