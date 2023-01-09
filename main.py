from settings import get_args_from_input
import wandb


tudataset_names = ["REDDIT-BINARY", "IMDB-BINARY", "MUTAG", "ENZYMES", "COLLAB", "PROTEINS"]
planetoid_names = ["cora", "citeseer", "pubmed"]
ogb_names = ["ogbg-molhiv"]
benchmark_names = ["TUDataset", "Planetoid", "ogb_graphproppred"]

def select_and_run_experiment(input_settings):
    """
    Run experiments on the selected benchmark or dataset.

    Args:
        input_settings (dict): Dictionary containing the input settings.
    """
    benchmark = ""
    if "dataset" in input_settings:
        dataset = input_settings["dataset"]
        if dataset in tudataset_names:
            benchmark = "TUDataset"
        elif dataset in planetoid_names:
            benchmark = "Planetoid"
        elif dataset in ogb_names:
            benchmark = "ogb_graphproppred"
        else:
            raise ValueError(f"Dataset \'{dataset}\' not found.")
    elif "benchmark" in input_settings:
        benchmark = input_settings["benchmark"]
    if benchmark == "TUDataset":
        from examples.tudataset import run
        run(input_settings)
    elif benchmark == "Planetoid":
        from examples.planetoid import run
        run(input_settings)
    elif benchmark == "ogb_graphproppred":
        from examples.ogb_graphproppred import run
        run(input_settings)
    else:
        raise ValueError("Benchmark not found.")

def select_and_run_tuning():
    wandb.init(group='v2')
    settings = {**dict(wandb.config), **get_args_from_input()}
    select_and_run_experiment(settings)

sweep_cfg = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'best_validation_criterion'
		},
    'parameters': {
        'learning_rate': {'values': [1e-3, 1e-2, 1e-1]},
        'num_hidden_layers': {'values': [2, 3, 4, 5]},
        'weight_decay': {'values': [1e-5, 1e-4, 1e-3, 1e-2]},
        'num_iterations': {'values': [0, 10, 20, 30, 40, 50]},
        'dataset': {'values': ['ENZYMES']},
        'rewiring': {'values': ['FoSR', 'SDRF', 'None']},
        'hidden_dim': {'values': [64, 128, 256, 512]},
        'dropout': {'values': [0.0, 0.1, 0.2, 0.3]},
        'layer_type': {'values': ['GCN', 'R-GCN', 'GIN', 'R-GIN']},
        'num_random_features': {'values': [0]},
     }
}

if __name__ == "__main__":
    input_settings = get_args_from_input()
    if "tuning" in input_settings and input_settings["tuning"] == True:
        # Run hyperparameter tuning.
        for i in range(100000):
            sweep_id = wandb.sweep(sweep_cfg, project="gnn2")
            wandb.agent(sweep_id, select_and_run_tuning)
    else:
        # Run experiment normally.
        select_and_run_experiment(input_settings)