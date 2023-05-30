from torch_geometric.datasets import TUDataset, Planetoid
from utils import train_val_test_split
from train import Experiment
from config.settings import parse_cfg

cfg = parse_cfg()
cfg.dataset = 'MUTAG'
cfg.task = 'graph_classification'
cfg.metric = 'accuracy'
cfg.goal = 'max'
cfg.input_dim = 7
cfg.output_dim = 2

mutag = TUDataset(root='data', name='MUTAG')
train_dataset, validation_dataset, test_dataset = train_val_test_split(mutag)


Experiment(cfg=cfg, train_dataset=train_dataset, validation_dataset=validation_dataset, test_dataset=test_dataset).run()