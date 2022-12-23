from torch_geometric.utils import to_networkx
from numba import jit
from typing import Tuple
import numpy as np
from math import inf

@jit(nopython=True)
def choose_edge_to_add(x: np.array,
                       edge_index: np.array,
                       degrees: np.array) -> Tuple[int, int]:
    """
        Chooses the edge (u, v) to add to the graph which minimizes y[u]*y[v].

        Args:
            x (np.array): Approximation of eigenvector.
            edge_index (np.array): Edge index of the graph in COO format.
            degrees (np.array): Node degrees.
        
        Returns:
            (u, v) (Tuple[int, int]): Edge to add.
    """
    num_nodes = x.size
    num_edges = edge_index.shape[1]
    y = x / ((degrees + 1) ** 0.5)
    products = np.outer(y, y)
    for i in range(num_edges):
        u = edge_index[0, i]
        v = edge_index[1, i]
        # Don't add existing edges.
        products[u, v] = inf
    for i in range(num_nodes):
        # Don't add self-loops.
        products[i, i] = inf
    smallest_product = np.argmin(products)
    return (smallest_product % num_nodes, smallest_product // num_nodes)

@jit(nopython=True)
def compute_degrees(edge_index: np.array,
                    num_nodes: int) -> np.array:
    """
        Computes the degrees of the nodes in the graph.

        Args:
            edge_index (np.array): Edge index of the graph in COO format.
            num_nodes (int): Number of nodes in the graph.

        Returns:
            degrees (np.array): Node degrees.
    """
    degrees = np.zeros(num_nodes)
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        degrees[edge_index[0, i]] += 1
    return degrees

@jit(nopython=True)
def add_edge(edge_index: np.array,
             u: int,
             v: int) -> np.array:
    """
        Adds an edge (u, v) to the graph.

        Args:
            edge_index (np.array): Edge index of the graph in COO format.
            u (int): Source node.
            v (int): Target node.

        Returns:
            new_edge_index (np.array): New edge index with edge (u, v) added.
    """
    new_edge = np.array([[u, v],[v, u]])
    return np.concatenate((edge_index, new_edge), axis=1)

@jit(nopython=True)
def adj_matrix_multiply(edge_index: np.array,
                        x: np.array) -> np.array:
    """
        Computes a matrix-vector product of the adjacency matrix of the graph.

        Args:
            edge_index (np.array): Edge index of the graph in COO format.
            x (np.array): Vector to multiply the adjacency matrix with.

        Returns:
            y (np.array): Result of the matrix-vector product.
    """
    num_nodes = x.size
    y = np.zeros(num_nodes)
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        u = edge_index[0, i]
        v = edge_index[1, i]
        y[u] += x[v]
    return y

@jit(nopython=True)
def _fosr(edge_index: np.array,
          edge_type: np.array,
          x: np.array,
          num_iterations: int,
          initial_power_iters: int) -> Tuple[np.array, np.array, np.array]:
    """
        Performs the FoSR algorithm. Returns the edge index of the graph after adding edges
        and the approximation of the eigenvector. Keeps track of the edge types (0 for original
        edges, 1 for added edges).

        Args:
            edge_index (np.array): Edge index of the graph in COO format.
            x (np.array): Approximation of eigenvector.
            num_iterations (int): Number of iterations to run the algorithm/edges to add.
            initial_power_iters (int): Number of power iterations to run before adding edges.

        Returns:
            edge_index (np.array): Edge index of the graph after adding edges.
            edge_type (np.array): Edge types of the graph after adding edges.
            x (np.array): Approximation of eigenvector after adding edges.
    """
    num_nodes = np.max(edge_index) + 1
    degrees = compute_degrees(edge_index, num_nodes=num_nodes)

    # Initial power iteration.
    for _ in range(initial_power_iters):
        x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
        y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
        x = y / np.linalg.norm(y)

    for _ in range(num_iterations):
        # Choose edge to add which minimizes y[u]*y[v].
        i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
        edge_index = add_edge(edge_index, i, j)
        degrees[i] += 1
        degrees[j] += 1
        edge_type = np.append(edge_type, 1)
        edge_type = np.append(edge_type, 1)

        # Power iteration.
        x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
        y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
        x = y / np.linalg.norm(y)
    return edge_index, edge_type, x

def fosr(edge_index: np.array=None,
         edge_type: np.array=None,
         x: np.array=None,
         num_iterations: int=50,
         initial_power_iters: int=5) -> Tuple[np.array, np.array, np.array]:
    """
        Wrapper for the FoSR algorithm. Returns the edge index of the graph after adding edges.
        
        Args:
            edge_index (np.array): Edge index of the graph in COO format.
            x (np.array): Approximation of eigenvector.
            num_iterations (int): Number of iterations to run the algorithm/edges to add.
            initial_power_iters (int): Number of power iterations to run before adding edges.

        Returns:
            edge_index (np.array): Edge index of the graph after adding edges.
            edge_type (np.array): Edge types of the graph after adding edges.
            x (np.array): Approximation of eigenvector after adding edges.
    """
    num_edges = edge_index.shape[1]
    num_nodes = np.max(edge_index) + 1
    if x is None:
        x = 2 * np.random.random(num_nodes) - 1
    if edge_type is None:
        edge_type = np.zeros(num_edges, dtype=np.int64)
    return _fosr(edge_index, edge_type=edge_type, x=x, num_iterations=num_iterations, initial_power_iters=initial_power_iters)