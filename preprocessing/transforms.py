from torch_geometric.transforms import BaseTransform
import torch
import torch_geometric

class AddRandomNodeFeatures(BaseTransform):
    """
    Adds or augments random node features to the graph.

    Args:
        num_features (int): Number of features to add.
    """
    def __init__(self, num_features: int=10):
        self.num_features = num_features

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        # Concatenate random features if data already has features, otherwise set features to random.
        if hasattr(data, "x") and data.x is not None:
            data.x = torch.cat([data.x, torch.randn(data.num_nodes, self.num_features)], dim=1)
        else:
            data.x = torch.randn(data.num_nodes, self.num_features)
        return data

class Rewire(BaseTransform):
    """
    Rewires the graph by adding edges.

    Args:
        num_iterations (int): Number of edges to add.
        rewiring_type (str): Type of rewiring to perform. One of "None", "SDRF", "FoSR".
    """
    def __init__(self, num_edges: int=1, rewiring_type: str="None"):
        self.num_edges = num_edges
        self.rewiring_type = rewiring_type

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        # Convert to networkx graph.
        graph = to_networkx(data, to_undirected=True)
        # Compute degrees.
        degrees = np.array([graph.degree[i] for i in range(graph.number_of_nodes())])
        # Add edges.
        for _ in range(self.num_edges):
            u, v = choose_edge_to_add(data.x.numpy(), data.edge_index.numpy(), degrees)
            graph.add_edge(u, v)
            degrees[u] += 1
            degrees[v] += 1
        # Convert back to PyTorch Geometric graph.
        data.edge_index = torch.tensor(np.array(graph.edges).T, dtype=torch.long)
        return data