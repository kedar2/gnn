import torch
import torch_geometric
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from experiment import Experiment
from torch_geometric.datasets import TUDataset
from models import GNN
from settings import Configuration, get_args_from_input
from preprocessing.transforms import Rewire, AddRandomFeaturesIfUnlabeled
import torch_geometric.transforms as T


class TUDatasetExperiment(Experiment):
    """
    An extension of the Experiment class for the TUDataset benchmark.

    Args:
        dataset (torch_geometric.data.Dataset): Full dataset.
        train_dataset (torch_geometric.data.Dataset): Dataset to be used for training.
        validation_dataset (torch_geometric.data.Dataset): Dataset to be used for validation.
        test_dataset (torch_geometric.data.Dataset): Dataset to be used for testing.
        cfg (Configuration): Configuration object containing experiment settings.
    """
    def __init__(self,
                dataset: torch_geometric.data.Dataset=None,
                train_dataset: torch_geometric.data.Dataset=None,
                validation_dataset: torch_geometric.data.Dataset=None,
                test_dataset: torch_geometric.data.Dataset=None,
                cfg: Configuration=None):
        super().__init__(dataset=dataset,
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        test_dataset=test_dataset,
                        cfg=cfg)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.model = GNN(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, loader: DataLoader):
        # Train loop
        self.model.train()
        for graph in tqdm(loader, disable=(not self.display)):
            self.optimizer.zero_grad()
            graph = graph.to(self.device)
            pred = self.model(graph)
            loss = self.loss_fn(input=pred, target=graph.y)
            loss.backward()
            self.optimizer.step()

    def eval(self, loader: DataLoader) -> float:
        # Evaluation loop, determines full-batch accuracy.
        self.model.eval()
        sample_size = len(loader.dataset)
        total_correct = 0
        with torch.no_grad():
            for graph in tqdm(loader, disable=(not self.display)):
                graph = graph.to(self.device)
                out = self.model(graph)
                _, pred = out.max(dim=1)
                total_correct += pred.eq(graph.y).sum().item()
        return total_correct / sample_size

def run():
    """
    Loads datasets, creates an experiment, and runs it.
    """

    # Assign default settings and overwrite with input settings.
    default_settings = {
    "display": True,
    "learning_rate": 1e-3,
    "max_epochs": 10000,
    "batch_size": 64,
    "eval_every": 1,
    "num_trials": 5,
    "dropout": 0.5,
    "weight_decay": 1e-5,
    "num_hidden_layers": 3,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "GCN",
    "rewiring": "FoSR",
    "num_iterations": 10,
    "num_relations": 2,
    "patience": 20,
    "dataset": "",
    "last_layer_fa": False,
    "lower_criterion_is_better": False,
    "task_type": "graph_classification"
    }

    dataset_settings = {
    "reddit": {
        "dataset": "reddit",
        "input_dim": 100,
        "output_dim": 2,
    },
    "imdb": {
        "dataset": "imdb",
        "input_dim": 100,
        "output_dim": 2,
    },
    "mutag": {
        "dataset": "mutag",
        "input_dim": 7,
        "output_dim": 2,
    },
    "enzymes": {
        "dataset": "enzymes",
        "input_dim": 3,
        "output_dim": 6,
    },
    "collab": {
        "dataset": "collab",
        "input_dim": 100,
        "output_dim": 2,
    },
    "proteins": {
        "dataset": "proteins",
        "input_dim": 3,
        "output_dim": 2}
    }
    
    input_settings = get_args_from_input()

    reddit = TUDataset(root="data", name="REDDIT-BINARY")
    imdb = TUDataset(root="data", name="IMDB-BINARY")
    mutag = TUDataset(root="data", name="MUTAG")
    enzymes = TUDataset(root="data", name="ENZYMES")
    collab = TUDataset(root="data", name="COLLAB")
    proteins = TUDataset(root="data", name="PROTEINS")

    dataset_names = {"reddit": reddit, "imdb": imdb, "mutag": mutag, "enzymes": enzymes, "collab": collab, "proteins": proteins}

    # Run experiment on all datasets or a single selected dataset.
    if "dataset" in input_settings:
        name = input_settings["dataset"]
        datasets = {name: dataset_names[name]}
    else:
        datasets = dataset_names
    
    for name, dataset in datasets.items():
        settings = {**default_settings, **dataset_settings[name], **input_settings}
        cfg = Configuration(**settings)

        # Preprocess dataset.
        transform = T.Compose([AddRandomFeaturesIfUnlabeled(100),
                                T.ToUndirected(),
                                Rewire(rewiring=cfg.rewiring,
                                        num_iterations=cfg.num_iterations)
                                ])
        dataset = [transform(graph) for graph in dataset]

        # Start experiment.
        print(f"Running experiment on {name}")
        for i in range(cfg.num_trials):
            print(f"Trial {i+1}")
            trial = wandb.init(project="gnn",
                        config={**default_settings, **dataset_settings[name], **input_settings},
                        group=f"{name}")
            experiment = TUDatasetExperiment(dataset=dataset, cfg=cfg)
            train_criterion, validation_criterion, test_criterion = experiment.run()

            trial.finish()
