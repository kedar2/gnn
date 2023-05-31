import wandb
import yaml
from run import main
import argparse

def single_experiment():
    """
    Runs a single experiment for a wandb agent.
    """
    wandb.init()
    cfg_dict = wandb.config.as_dict()
    cfg_dict["wandb"] = True
    cfg_dict["sweep"] = True
    main(cfg_dict)

def sweep(cfg_dict: str):
    """
    Performs a hyperparameter sweep over the configurations specified in the given config dictionary.
    """
    sweep_config = {"parameters": cfg_dict, "method": "grid"}
    sweep_id = wandb.sweep(sweep_config, project="gnn")
    wandb.agent(sweep_id, function=single_experiment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default="config/sweeps/tudataset.yml")
    cfg_file = parser.parse_args().cfg_file

    cfg_dict = yaml.load(open(cfg_file, 'r'), Loader=yaml.FullLoader)
    
    for key, value in cfg_dict.items():
        cfg_dict[key] = {"values": value}
    sweep(cfg_dict)