from torch_geometric.transforms import Constant
import torch, torch_geometric, torch_geometric.loader
from torch_geometric.loader import DataLoader
import yaml
from typing import Optional
from tqdm import tqdm
from config import ExperimentConfig
from preprocess import load_experiment



def main(cfg_dict: dict = None):
    """
    Runs an experiment with the given configuration.

    Args:
        cfg_dict (dict): Dictionary containing the settings for the experiment.
    """
    dataset, cfg, split = load_experiment(cfg_dict)
    if "graph" in cfg.task:
        from exp.graph_task import run_experiment
        run_experiment(dataset, split, cfg)
    elif "node" in cfg.task:
        graph = dataset.data
        from exp.node_task import run_experiment
        run_experiment(graph, split, cfg)
    else:
        raise NotImplementedError(f"Task {cfg.task} not implemented.")


    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sample.yaml", help="path to config file")
    args = parser.parse_args()
    cfg_dict = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(cfg_dict)