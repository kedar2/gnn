import torch
import torch_geometric
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from experiment import Experiment
from models import GNN
from settings import Configuration, get_args_from_input
from preprocessing.transforms import Rewire, AddRandomNodeFeatures, AddOneFeatures
import torch_geometric.transforms as T



class TUDatasetExperiment(Experiment):
    """
    An extension of the Experiment class for the OGB benchmark.

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
        cfg.input_dim = dataset[0].x.shape[1]
        self.model = GNN(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

    def train(self, loader: DataLoader):
        # Train loop    
        self.model.train()
        total_loss = 0
        for graph in tqdm(loader, disable=(not self.display)):
            self.optimizer.zero_grad()
            graph = graph.to(self.device)
            pred = self.model(graph)
            loss = self.loss_fn(input=pred, target=graph.y)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step(total_loss)

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

#TODO: add virtual node setting, standardize pipeline across benchmarks

def run(input_settings: dict={}):
    """
    Loads datasets, creates an experiment, and runs it.

    Args:
        input_settings (dict): Dictionary containing settings to overwrite default settings.
    """

    # Assign default settings and overwrite with input settings.
    default_settings = {
    "display": True,
    "learning_rate": 1e-3,
    "max_epochs": 10000,
    "batch_size": 64,
    "eval_every": 1,
    "num_trials": 1,
    "dropout": 0.5,
    "weight_decay": 0,
    "num_hidden_layers": 4,
    "num_random_features": 0,
    "hidden_dim": 64,
    "layer_type": "GCN",
    "rewiring": "FoSR",
    "num_iterations": 10,
    "num_relations": 2,
    "patience": 100,
    "dataset": "",
    "last_layer_fa": False,
    "lower_criterion_is_better": False,
    "task_type": "graph_classification"
    }

    dataset_settings = {
    "ogbg-molhiv": {
        "dataset": "ogbg_molhiv",
        "output_dim": 2,
    }
    }

    molhiv = PygGraphPropPredDataset(name="ogbg-molhiv", root='data')

    dataset_names = {"ogbg-molhiv": molhiv}
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
        transform = T.Compose([AddOneFeatures(),
                                AddRandomNodeFeatures(cfg.num_random_features),
                                T.ToUndirected(),
                                Rewire(rewiring=cfg.rewiring,
                                        num_iterations=cfg.num_iterations)
                                ])
        dataset = [transform(graph) for graph in dataset]

        # Start experiment.
        print(f"Running experiment on {name}")
        for i in range(cfg.num_trials):
            print(f"Trial {i+1}")

            if cfg.wandb and not cfg.tuning:
                # Initialize wandb, not necessary if already initialized for tuning.
                trial = wandb.init(project="gnn",
                        config={**default_settings, **dataset_settings[name], **input_settings},
                        group=f"{name}")

            experiment = TUDatasetExperiment(dataset=dataset, cfg=cfg)
            experiment.run()

            if cfg.wandb and not cfg.tuning:
                # Finish wandb run, not necessary if performing further tuning.
                trial.finish()