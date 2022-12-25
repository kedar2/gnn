import torch
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GINConv, FiLMConv, global_mean_pool
from settings import Configuration

class RGINConv(torch.nn.Module):
    r"""
    The relational GIN convolutional layer from the
    'GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation <https://arxiv.org/abs/1812.08797>' paper.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_relations (int): Number of relations.    
    """
    def __init__(self,
                in_features: int,
                out_features: int,
                num_relations: int):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for _ in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class GNN(torch.nn.Module):
    """
    A generic GNN model.

    Args:
        cfg (Configuration): Configuration object containing experiment settings.
    """
    def __init__(self,
                 cfg: Configuration=None):
        super(GNN, self).__init__()
        self.num_relations = cfg.num_relations
        self.layer_type = cfg.layer_type
        num_features = [cfg.input_dim] + [cfg.hidden_dim] * cfg.num_hidden_layers + [cfg.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for (in_features, out_features) in zip(num_features[:-1], num_features[1:]):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)

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
        if "graph" in cfg.task_type:
            self.pooling = True
        else:
            self.pooling = False

    def get_layer(self, in_features: int, out_features: int) -> torch.nn.Module:
        """
        Returns a layer of the specified type.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)

    def forward(self, graph) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GIN"]:
                if not hasattr(graph, "edge_type"):
                    raise ValueError("Graphs must have an edge_type attribute for R-GCN and R-GIN layers.")
                x_new = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x_new = layer(x, edge_index)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            if i == self.num_layers - 1 and self.last_layer_fa:
                # Handle final layer if it is fully adjacent.
                combined_values = global_mean_pool(x, graph.batch)
                combined_values = self.last_layer_transform(combined_values)
                if self.layer_type in ["R-GCN", "R-GIN"]:
                    x_new = x_new + combined_values[graph.batch]
                else:
                    x_new = combined_values[graph.batch]
            x = x_new
        if self.pooling:
            x = global_mean_pool(x, graph.batch)
        return x