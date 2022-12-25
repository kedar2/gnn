import torch
import torch_geometric
import wandb
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from math import inf
from settings import Configuration

class Experiment:
    """
    Experiment class for executing training and testing loops.

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
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.train_fraction = cfg.train_fraction
        self.validation_fraction = cfg.validation_fraction
        self.test_fraction = cfg.test_fraction
        self.max_epochs = cfg.max_epochs
        self.lower_criterion_is_better = cfg.lower_criterion_is_better
        self.device = cfg.device
        self.display = cfg.display
        self.patience = cfg.patience
        self.eval_every = cfg.eval_every

        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # If the dataset does not already have a train/validation/test split, randomly assign one.
        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.train_fraction * dataset_size)
            validation_size = int(self.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset,[train_size, validation_size, test_size])
        
    def run(self):
        # Keep track of progress to determine when to stop.
        if self.lower_criterion_is_better:
            best_validation_criterion = inf
        else:
            best_validation_criterion = -inf
        epochs_no_improve = 0

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(1, 1 + self.max_epochs):

            # Train for one epoch.
            if self.display:
                print(f"====Epoch {epoch}\nTraining...")
            self.train(loader=train_loader)
            
            new_best_str = ''
            if epoch % self.eval_every == 0:
                # Evaluate, report progress, and check whether to stop training.
                if self.display:
                    print(f"\nEvaluating...")
                train_criterion= self.eval(loader=train_loader)
                validation_criterion = self.eval(loader=validation_loader)
                test_criterion = self.eval(loader=test_loader)

                if self.cfg.wandb:
                    wandb.log({'train_criterion': train_criterion, 'validation_criterion': validation_criterion, 'test_criterion': test_criterion})

                if ((validation_criterion < best_validation_criterion * 0.9999) and self.lower_criterion_is_better) or ((validation_criterion > best_validation_criterion * 1.0001) and not self.lower_criterion_is_better):
                    # Checks if the validation loss is the best so far.
                    # best_train_criterion and best_test_criterion are defined as the train and test loss when the validation loss is the lowest.
                    best_train_criterion = train_criterion
                    best_validation_criterion = validation_criterion
                    best_test_criterion = test_criterion
                    epochs_no_improve = 0
                    new_best_str = ' (new best validation)'
                else:
                    epochs_no_improve += 1
                if self.display:
                    print(f'\nTrain: {train_criterion}, Validation: {validation_criterion}{new_best_str}, Test: {test_criterion}\n')
                if epochs_no_improve > self.patience:
                    if self.display:
                        print(f'\n{self.patience} epochs without improvement, stopping training\n')
                        print(f'Best train: {best_train_criterion}, Best validation: {best_validation_criterion}, Best test: {best_test_criterion}')
                    if self.cfg.wandb:
                        wandb.log({'best_train_criterion': best_train_criterion, 'best_validation_criterion': best_validation_criterion, 'best_test_criterion': best_test_criterion})
                    return best_train_criterion, best_validation_criterion, best_test_criterion
        if self.display:
            print('\nReached max epoch count, stopping training')
            if self.cfg.wandb:
                wandb.log({'best_train_criterion': best_train_criterion, 'best_validation_criterion': best_validation_criterion, 'best_test_criterion': best_test_criterion})
            print(f'Best train acc: {best_train_criterion}, Best validation loss: {best_validation_criterion}, Best test loss: {best_test_criterion}')

        return train_criterion, validation_criterion, test_criterion

