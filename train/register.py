from models.gnn import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss

model_register = {
    "GNN": GNN 
}

scheduler_register = {
    None: None,
    "ReduceLROnPlateau": ReduceLROnPlateau
}

optimizer_register = {
    "Adam": Adam
}

loss_fn_register = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "MSELoss": MSELoss
}
