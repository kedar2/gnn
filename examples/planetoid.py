import torch
import torch_geometric
from tqdm import tqdm
from torch.utils.data import DataLoader
from train import Experiment
from torch_geometric.datasets import Planetoid
from models.gnn import GNN
from config.parsing import Configuration, get_args_from_input


class PlanetoidExperiment(Experiment):
    """
    An extension of the Experiment class for the Planetoid datasets.
    Since the task is node classification, train/validation/test indices are required (instead of datasets).

    Args:
        dataset (torch_geometric.data.Dataset): Full dataset.
        train_index (torch.tensor): Index of nodes to be used for training.
        validation_index (torch.tensor): Index of nodes to be used for validation.
        test_index (torch.tensor): Index of nodes to be used for testing.
        cfg (Configuration): Configuration object containing experiment settings.
    """
    def __init__(self,
                dataset: torch_geometric.data.Dataset=None,
                train_index: torch.tensor=None,
                validation_index: torch.tensor=None,
                test_index: torch.tensor=None,
                cfg: Configuration=None):

        # Index of all nodes in the graph.
        self.index = torch.arange(0, dataset.data.num_nodes)
        self.graph = dataset.data

        super().__init__(dataset=self.index,
                        train_dataset=train_index,
                        validation_dataset=validation_index,
                        test_dataset=test_index,
                        cfg=cfg)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.x = self.graph.x.to(self.device)
        self.y = self.graph.y.to(self.device)

        # Set train, validation, and test indices after the split has been assigned.
        self.train_index = self.train_dataset
        self.validation_index = self.validation_dataset
        self.test_index = self.test_dataset

        self.model = GNN(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, loader: DataLoader):
        # Train loop
        self.model.train()
        for index in tqdm(loader, disable=(not self.display)):
            self.optimizer.zero_grad()
            pred = self.model(self.graph)
            loss = self.loss_fn(input=pred[index], target=self.y[index])
            loss.backward()
            self.optimizer.step()

    def eval(self, loader: DataLoader) -> float:
        # Evaluation loop, determines full-batch accuracy.
        self.model.eval()
        sample_size = len(loader.dataset)
        total_correct = 0
        out = self.model(self.graph)
        _, pred = out.max(dim=1)
        with torch.no_grad():
            for index in loader:
                total_correct += pred[index].eq(self.y[index]).sum().item()
        return total_correct / sample_size

def run(input_settings: dict={}):
    """
    Loads datasets, creates an experiment, and runs it.

    Args:
        input_settings (dict): Dictionary containing settings to overwrite the default settings.
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
    "num_hidden_layers": 1,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "GCN",
    "rewiring": "fosr",
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 20,
    "dataset": "",
    "last_layer_fa": False,
    "task_type": "node_classification",
    "lower_criterion_is_better": False
    }

    dataset_settings = {
    "cora": {
        "dataset": "cora",
        "input_dim": 1433,
        "output_dim": 7,
    },
    "citeseer": {
        "dataset": "citeseer",
        "input_dim": 3703,
        "output_dim": 6,
    },
    "pubmed": {
        "dataset": "pubmed",
        "input_dim": 500,
        "output_dim": 3,
    },
    }
    

    cora = Planetoid(root="data", name="cora")
    citeseer = Planetoid(root="data", name="citeseer")
    pubmed = Planetoid(root="data", name="pubmed")
    dataset_names = {"cora": cora, "citeseer": citeseer, "pubmed": pubmed}

    # Run experiment on all datasets or a single selected dataset.
    if "dataset" in input_settings:
        name = input_settings["dataset"]
        datasets = {name: dataset_names[name]}
    else:
        datasets = dataset_names
    
    for name, dataset in datasets.items():
        settings = {**default_settings, **dataset_settings[name], **input_settings}
        cfg = Configuration(**settings)
        print(f"Running experiment on {name}")
        for i in range(cfg.num_trials):
            print(f"Trial {i+1}")
            experiment = PlanetoidExperiment(dataset=dataset, cfg=cfg)
            train_criterion, validation_criterion, test_criterion = experiment.run()