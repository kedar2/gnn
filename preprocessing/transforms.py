from torch_geometric.transforms import BaseTransform
from preprocessing.fosr import fosr
from preprocessing.sdrf import sdrf
import torch
import torch_geometric


class Rewire(BaseTransform):
    """
    Rewires the graph by adding edges.

    Args:
        num_iterations (int): Number of edges to add.
        rewiring (str): Type of rewiring to perform. One of "None", "SDRF", "FoSR".
    """
    def __init__(self, num_iterations: int=10, rewiring: str="FoSR"):
        self.num_iterations = num_iterations
        self.rewiring = rewiring
    
    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """
        Rewires the graph by adding edges.

        Args:
            data (torch_geometric.data.Data): Graph data.
        
        Returns:
            data (torch_geometric.data.Data): Graph data with rewired edges.
        
        Raises:
            ValueError: If the rewiring type is not recognized.
        """
        if self.rewiring == "FoSR":
            edge_index, edge_type, _ = fosr(edge_index=data.edge_index.numpy(),
                                            num_iterations=self.num_iterations)
            data.edge_index = torch.from_numpy(edge_index)
            data.edge_type = torch.from_numpy(edge_type)
            return data
        elif self.rewiring == "SDRF":
            edge_index, edge_type = sdrf(data=data, loops=self.num_iterations)
            data.edge_index = edge_index
            data.edge_type = edge_type
            return data
        elif self.rewiring == "None":
            # Add edge type attribute
            data.edge_type = torch.zeros(data.edge_index.shape[1])
            return data
        else:
            raise ValueError(f"Rewiring type {self.rewiring} not recognized.")

class AddRandomNodeFeatures(BaseTransform):
    """
    Adds or augments random node features to the graph.

    Args:
        num_features (int): Number of features to add.
    """
    def __init__(self, num_features: int=10):
        self.num_features = num_features

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """
        Concatenate random features if data already has features, otherwise set features to random.

        Args:
            data (torch_geometric.data.Data): Graph data.
        
        Returns:
            data (torch_geometric.data.Data): Graph data with random features.
        """
        if hasattr(data, "x") and data.x is not None:
            data.x = torch.cat([data.x, torch.randn(data.num_nodes, self.num_features)], dim=1)
        else:
            data.x = torch.randn(data.num_nodes, self.num_features)
        return data        

class AddRandomFeaturesIfUnlabeled(BaseTransform):
    """
    Adds random node features to the graph if the nodes are unlabeled.

    Args:
        num_features (int): Number of features to add.
    """
    def __init__(self, num_features: int=10):
        self.num_features = num_features

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """
        Does nothing if data already has features, otherwise set random features.

        Args:
            data (torch_geometric.data.Data): Graph data.
        
        Returns:
            data (torch_geometric.data.Data): Graph data with random features (if applicable).
        """
        if hasattr(data, "x") and data.x is not None:
            pass
        else:
            data.x = torch.randn(data.num_nodes, self.num_features)
        return data

