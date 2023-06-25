import torch
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GINConv, FiLMConv, global_mean_pool
from config import ExperimentConfig

class GNN(torch.nn.Module):
    """
    A generic GNN model.

    Args:
        cfg (ExperimentConfig): Configuration object containing experiment settings.
    """
    def __init__(self,
                 cfg: ExperimentConfig=None):
        super(GNN, self).__init__()
        self.num_relations = 1
        self.layer_type = cfg.layer_type
        self.residual_connections = cfg.residual_connections
        num_features = [cfg.input_dim] + [cfg.hidden_dim] * cfg.num_hidden_layers + [cfg.output_dim]
        self.num_layers = len(num_features) - 1
        self.layers = ModuleList()
        for (in_features, out_features) in zip(num_features[:-1], num_features[1:]):
            self.layers.append(self.get_layer(in_features, out_features))
        if self.residual_connections:
            self.residual_layers = ModuleList()
            for (in_features, out_features) in zip(num_features[:-1], num_features[1:]):
                self.residual_layers.append(torch.nn.Linear(in_features, out_features))

        self.dropout = Dropout(p=cfg.dropout)
        self.act_fn = ReLU()
        self.last_layer_fa = cfg.last_layer_fa

        if cfg.last_layer_fa:
            # Add transformation associated with complete graph if last layer is fully adjacent
            if "GCN" in cfg.layer_type:
                self.last_layer_transform = torch.nn.Linear(cfg.hidden_dim, cfg.output_dim)
            elif "GIN" in cfg.layer_type:
                self.last_layer_transform = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
                                                            nn.BatchNorm1d(self.args.hidden_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(self.args.hidden_dim, self.args.output_dim))
            else:
                raise NotImplementedError(f"Last layer fully adjacent is not implemented for layer type {cfg.layer_type}.")

        # Pooling is only used for graph tasks
        if "graph" in cfg.task:
            self.pooling = True
        else:
            self.pooling = False

    def get_layer(self, in_features: int, out_features: int) -> torch.nn.Module:
        """
        Returns a layer of the specified type.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        Returns:
            torch.nn.Module: Layer of the specified type.
        """
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
        else:
            raise ValueError(f"Layer type {self.layer_type} not supported.")

    def forward(self, graph) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        for i, layer in enumerate(self.layers):
            x_new = layer(x, edge_index)
            if self.residual_connections:
                x_new = x_new + self.residual_layers[i](x)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            if i == self.num_layers - 1 and self.last_layer_fa:
                # Handle final layer if it is fully adjacent.
                combined_values = global_mean_pool(x, graph.batch)
                combined_values = self.last_layer_transform(combined_values)
                x_new = combined_values[graph.batch]
            x = x_new
        if self.pooling:
            x = global_mean_pool(x, graph.batch)
        return x