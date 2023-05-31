from utils import train_val_test_split
from train import Experiment
from config.parsing import parse_cfg

def main():
    cfg = parse_cfg()
    cfg.set_defaults()
    print(cfg.asdict())
    input()
    if cfg.dataset == "MUTAG":
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='data', name='MUTAG')
        train_dataset, validation_dataset, test_dataset = train_val_test_split(dataset)
        Experiment(cfg=cfg, train_dataset=train_dataset, validation_dataset=validation_dataset, test_dataset=test_dataset).run()

if __name__ == "__main__":
    main()