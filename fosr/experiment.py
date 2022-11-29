import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from math import inf
from attrdict import AttrDict

class Experiment:
    def __init__(self, args=None, dataset=None, train_dataset=None, validation_dataset=None, test_dataset=None):

        default_args = AttrDict({
            "display": True,
            "dataset": None,
            "train_dataset": None,
            "validation_dataset": None,
            "test_dataset": None,
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
            "max_epochs": 1000000,
            "logs": None,
            "lower_loss_is_better": True,
            "device": None
        }
        )

        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        # randomly assign a train/validation/test split, or train/validation split if test already assigned
        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.args.train_fraction * dataset_size)
            validation_size = int(self.args.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset,[train_size, validation_size, test_size])
        elif self.validation_dataset is None:
            train_size = int(self.args.train_fraction * len(self.train_dataset))
            validation_size = len(self.args.train_data) - train_size
            self.args.train_data, self.args.validation_data = random_split(self.args.train_data, [train_size, validation_size])
        
    def run(self):

        if self.args.display:
            print("Starting training")

        # Keep track of progress to determine when to stop
        if self.args.lower_loss_is_better:
            best_validation_loss = inf
        else:
            best_validation_loss = -inf
        epochs_no_improve = 0

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(1, 1 + self.args.max_epochs):
            
            self.train(loader=train_loader)

            new_best_str = ''
            if epoch % self.args.eval_every == 0:
                # Evaluate, report progress, and check whether to stop training
                train_loss= self.eval(loader=train_loader)
                validation_loss = self.eval(loader=validation_loader)
                test_loss = self.eval(loader=test_loader)

                if ((validation_loss < best_validation_loss * 0.9999) and self.args.lower_loss_is_better) or ((validation_loss > best_validation_loss * 1.0001) and not self.args.lower_loss_is_better):
                    # Checks if the validation loss is the best so far.
                    # Note: best_train_loss and best_test_loss are defined as the train and test loss when the validation loss is the lowest.
                    best_train_loss = train_loss
                    best_validation_loss = validation_loss
                    best_test_loss = test_loss
                    epochs_no_improve = 0
                    new_best_str = ' (new best validation)'
                else:
                    epochs_no_improve += 1
                if self.args.display:
                    print(f'Epoch {epoch}, Train: {train_loss}, Validation: {validation_loss}{new_best_str}, Test: {test_loss} {new_best_str}')
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train: {best_train_loss}, Best validation: {best_validation_loss}, Best test: {best_test_loss}')
                    return best_train_loss, best_validation_loss, best_test_loss
        if self.args.display:
            print('Reached max epoch count, stopping training')
            print(f'Best train acc: {best_train_loss}, Best validation loss: {best_validation_loss}, Best test loss: {best_test_loss}')

        return train_loss, validation_loss, test_loss
