from fosr.experiment import Experiment
from attrdict import AttrDict
from examples.planetoid.models import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from tqdm import tqdm

class PlanetoidExperiment(Experiment):
    def __init__(self, args=None, dataset=None, train_mask=None, validation_mask=None, test_mask=None):

        self.mask = list(range(dataset.data.num_nodes))

        # Instead of using train/validation/test datasets, we just use masks, since we are doing node classification on a single graph.
        super(PlanetoidExperiment, self).__init__(args=args, dataset=self.mask, train_dataset=train_mask, validation_dataset=validation_mask, test_dataset=test_mask)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.graph = dataset.data
        self.x = self.graph.x.to(self.args.device)
        self.y = self.graph.y.to(self.args.device)
        self.args.input_dim = self.x.shape[1]
        self.args.output_dim = torch.amax(self.y).item() + 1

        
        self.model = GNN(self.args).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)
    
    def train(self, loader):
        self.model.train()
        self.optimizer.zero_grad()
        for mask in tqdm(loader, disable=(not self.args.display)):
            pred = self.model(self.graph)
            loss = self.loss_fn(input=pred[mask], target=self.y[mask])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

    def eval(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        total_correct = 0
        out = self.model(self.graph)
        _, pred = out.max(dim=1)
        with torch.no_grad():
            for mask in loader:
                total_correct += pred[mask].eq(self.y[mask]).sum().item()
        return total_correct / sample_size