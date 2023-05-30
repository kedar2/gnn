import torch
import torch_geometric
from tqdm import tqdm
from torch.utils.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from train import Experiment
from models.gnn import GNN
from config.settings import Configuration, get_args_from_input
from preprocessing.transforms import Rewire, AddRandomNodeFeatures, AddOneFeatures
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch_geometric.transforms as T

class FeatureEmbedding(torch.nn.Module):
    """
    Module for embedding categorical node and edge features before passing them to the GNN.
    """
    def __init__(self, emb_dim: int=100):
        super(FeatureEmbedding, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(self, graph: torch_geometric.data.Data) -> torch_geometric.data.Data:
        graph.x = self.atom_encoder(graph.x)
        return graph

class OGBGraphPropPredExperiment(Experiment):
    """
    An extension of the Experiment class for the OGB graph property prediction benchmark.

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
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model = torch.nn.Sequential(
            FeatureEmbedding(emb_dim=cfg.input_dim),
            GNN(cfg)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.evaluator = Evaluator(name=cfg.dataset)

    def train(self, loader: DataLoader):
        # Train loop    
        self.model.train()
        total_loss = 0
        for graph in tqdm(loader, disable=(not self.display)):
            self.optimizer.zero_grad()
            graph = graph.to(self.device)
            pred = self.model(graph)
            loss = self.loss_fn(input=pred, target=graph.y.float())
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step(total_loss)

    def eval(self, loader: DataLoader):
        # Evaluation loop
        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for graph in tqdm(loader):
                graph = graph.to(self.device)

                y_true_batch = graph.y.to(self.device).view(-1)
                y_true += (y_true_batch.tolist())

                y_pred_batch = self.model(graph)

                y_pred += (y_pred_batch.tolist())
        y_true = torch.tensor(y_true).view(-1, 1)
        y_pred = torch.tensor(y_pred).view(-1, 1)
        return self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})['rocauc']

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
    "input_dim": 100,
    "num_hidden_layers": 4,
    "num_random_features": 0,
    "hidden_dim": 64,
    "layer_type": "R-GCN",
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
        "output_dim": 1,
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

            experiment = OGBGraphPropPredExperiment(dataset=dataset, cfg=cfg)
            experiment.run()