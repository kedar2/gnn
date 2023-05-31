from utils import train_val_test_split
from train import Experiment
from config.parsing import parse_settings, Configuration
from torch_geometric.transforms import Constant

def main(args: dict=None):
    cfg = Configuration(**args)
    cfg.set_defaults()
    if args:
        cfg.update(args)
    if cfg.dataset in ["REDDIT-BINARY", "IMDB-BINARY", "COLLAB", "ENZYMES", "PROTEINS", "MUTAG"]:
        from torch_geometric.datasets import TUDataset

        # add constant node features for datasets without node features
        if cfg.dataset in ["REDDIT-BINARY", "IMDB-BINARY", "COLLAB"]:
            dataset = TUDataset(root='data', name=cfg.dataset, pre_transform=Constant())
        else:
            dataset = TUDataset(root='data', name=cfg.dataset)

        train_dataset, validation_dataset, test_dataset = train_val_test_split(dataset)
        Experiment(cfg=cfg, train_dataset=train_dataset, validation_dataset=validation_dataset, test_dataset=test_dataset).run()

if __name__ == "__main__":
    args = parse_settings()
    main(args)