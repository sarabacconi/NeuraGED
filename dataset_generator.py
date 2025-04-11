# %%
from typing import Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

# %%
# Parameters for dataset generation
feature_dim = 27  # Feature vector dimension (as in ZINC dataset)
mean_n_nodes_first_graph = 23  # Mean number of nodes for the first graph
variance_n_nodes_first_graph = 3  # Variance in node count
max_n_nodes_second_graph = 38  # Max number of nodes in the second graph
half_n_couples = 500  # Dataset will contain 2 * this many pairs
features_init_ru = "random_uniform"  # Feature initialization strategy
features_init_rn = "random_normal"
features_init_1 = "ones"
padding_value = -1  # Value used for padding features and adjacency matrices


# %%
# Convert adjacency matrix to a NetworkX graph object
def build_graph_structure(adjacency_matrix):
    G = nx.Graph()
    n_node = adjacency_matrix.shape[0]
    for i in range(n_node):
        for j in range(i):
            if adjacency_matrix[i, j]:
                G.add_edge(i, j)
    return G

# Draw a graph with spring layout
def draw_graph(G):
    random_pos = nx.random_layout(G, seed=1)
    pos = nx.spring_layout(G, pos=random_pos)
    nx.draw(G, pos, with_labels=True)
    plt.show()



# %%
# Generate a random adjacency matrix for the first graph
def generate_first_graph(n_nodes):
    K = n_nodes
    A = abs(np.random.normal(0, size=(K, K)))  # Random weights
    m = A < 1
    A[m] = 0
    A[~m] = 1
    A = np.sign(A + A.T)  # Symmetric binary matrix
    np.fill_diagonal(A, 0)
    A = A.astype(int)

    # Ensure all nodes have at least one connection
    if not np.all(A.sum(axis=1)) and K > 1:
        idx = np.nonzero(A.sum(axis=1) == 0)[0]
        for i in idx:
            if i < len(A) - 1:
                j = np.random.randint(i + 1, len(A))
            else:
                j = np.random.randint(0, len(A) - 1)
            A[i][j] = A[j][i] = 1

    return A


# Generate a graph pair (G1 and G2) where G2 extends G1 with new nodes and edges
def generate_coupled_graphs(n_nodes, n_node_to_add):
    A = generate_first_graph(n_nodes)

    if n_node_to_add == 0:
        G1, G2 = build_graph_structure(A), build_graph_structure(A)
        ged_not_normalized = ged_vertex_normalized = ged_vertex_and_edges_normalized = 0
    else:
        # Generate new adjacency matrix with additional nodes
        A1 = abs(np.random.normal(0, size=(A.shape[0] + n_node_to_add, A.shape[1] + n_node_to_add)))
        m = A1 < 1
        A1[m] = 0
        A1[~m] = 1
        A1 = np.sign(A1 + A1.T)
        np.fill_diagonal(A1, 0)

        # Copy original adjacency into top-left corner
        A1[:A.shape[0], :A.shape[1]] = A
        A = A.astype(int)
        A1 = A1.astype(int)

        # Ensure all new nodes are connected
        if len(A1) > 1 and not np.all(A1.sum(axis=1)):
            idx = np.nonzero(A1.sum(axis=1) == 0)[0]
            for i in idx:
                if i < len(A1) - 1:
                    j = np.random.randint(i + 1, len(A1))
                else:
                    j = np.random.randint(0, len(A1) - 1)
                A1[i][j] = A1[j][i] = 1

        G1 = build_graph_structure(A)
        G2 = build_graph_structure(A1)

        # Estimate non-normalized and normalized GEDs
        ged_not_normalized = np.sum(A1[-n_node_to_add:, :n_nodes]) + np.sum(A1[-n_node_to_add:, -n_node_to_add:]) / 2 + abs(n_node_to_add)
        ged_vertex_normalized = ged_not_normalized / (len(G1.nodes) + len(G2.nodes))
        ged_vertex_and_edges_normalized = ged_not_normalized / (len(G1.nodes) + len(G2.nodes) + G1.number_of_edges() + G2.number_of_edges())

    return G1, G2, ged_not_normalized, ged_vertex_normalized, ged_vertex_and_edges_normalized

# %%
# Class that builds a pair of graphs and extracts structural info
class CoupleGenerator:
    def __init__(self, n_nodes: int, n_node_to_add: int):
        self.n_nodes = n_nodes
        self.n_node_to_add = n_node_to_add

    def __call__(self, features_init):
        G_l, G_r, ged_n, ged_v, ged_ve = generate_coupled_graphs(self.n_nodes, self.n_node_to_add)

        # Estimate diameter
        if nx.is_connected(G_l) and nx.is_connected(G_r):
            diam = max(nx.diameter(G_l), nx.diameter(G_r))
        elif nx.is_connected(G_l):
            diam = nx.diameter(G_l)
        elif nx.is_connected(G_r):
            diam = nx.diameter(G_r)
        else:
            diam = 0

        # Feature matrix initialization
        x_l = self.init_features_matrix(G_l.number_of_nodes(), feature_dim, torch.tensor(nx.to_numpy_array(G_l)), features_init)
        x_r = self.init_features_matrix(G_r.number_of_nodes(), feature_dim, torch.tensor(nx.to_numpy_array(G_r)), features_init)

        return Data(x=x_l, adj=torch.tensor(nx.to_numpy_array(G_l))), Data(x=x_r, adj=torch.tensor(nx.to_numpy_array(G_r))), ged_n, ged_v, ged_ve, diam

    def init_features_matrix(self, num_node, input_dim, adj, features_init):
        if features_init == "ones":
            return torch.ones([num_node, input_dim])
        elif features_init == "random_uniform":
            return torch.rand((num_node, input_dim))
        elif features_init == "random_normal":
            return torch.randn((num_node, input_dim))
        else:
            raise NotImplementedError

# %%
# Function to generate the full dataset of graph pairs
def generate_dataset(n_graphs, features_init):
    data = []
    max_diameter_l_r = []
    n_nodes_all_graphs = []
    ged_list_not_normalized, ged_list_normalized_vertex, ged_list_normalized_vertex_edges = [], [], []

    for i in range(n_graphs):
        n_nodes = int(np.random.normal(mean_n_nodes_first_graph, variance_n_nodes_first_graph))
        n_node_to_add = np.random.randint(0, max_n_nodes_second_graph - n_nodes) if n_nodes < max_n_nodes_second_graph else 0

        Data_i = CoupleGenerator(n_nodes, n_node_to_add)
        triplet_data = Data_i(features_init)

        n_nodes_all_graphs.append(max(triplet_data[0].num_nodes, triplet_data[1].num_nodes))
        data.append([triplet_data[0], triplet_data[1], triplet_data[2]])
        ged_list_not_normalized.append(triplet_data[2])
        max_diameter_l_r.append(triplet_data[5])
        ged_list_normalized_vertex.append(triplet_data[3])
        ged_list_normalized_vertex_edges.append(triplet_data[4])

    max_n_nodes = max(n_nodes_all_graphs)
    max_min_ged_not_normalized = [max(ged_list_not_normalized), min(ged_list_not_normalized)]
    max_diam = max(max_diameter_l_r)

    return data, max_n_nodes, max_min_ged_not_normalized, max_diam

# %%
# Apply padding and masking to graph pairs
def padding_and_masking_data(data, max_total):
    data_list = []
    data_list_swapped = []

    for i in range(len(data)):
        padding_size_l = max_total - data[i][0].num_nodes
        padding_size_r = max_total - data[i][1].num_nodes

        # Pad feature matrices and adjacency matrices
        x_i_l = F.pad(data[i][0].x, (0, 0, 0, padding_size_l), "constant", padding_value).to(torch.float32)
        adj_i_l = F.pad(data[i][0].adj, (0, padding_size_l, 0, padding_size_l), "constant", padding_value).to(torch.float32)

        x_i_r = F.pad(data[i][1].x, (0, 0, 0, padding_size_r), "constant", padding_value).to(torch.float32)
        adj_i_r = F.pad(data[i][1].adj, (0, padding_size_r, 0, padding_size_r), "constant", padding_value).to(torch.float32)

        # Create masks (1 where data is valid)
        mask_l = (x_i_l[:, :1] != padding_value) * 1
        mask_r = (x_i_r[:, :1] != padding_value) * 1

        # Create normal and swapped data entries
        data_list.append(Data(x_l=x_i_l, adj_l=adj_i_l, mask_l=mask_l, x_r=x_i_r, adj_r=adj_i_r, mask_r=mask_r,
                              y=torch.tensor(data[i][2]).to(torch.float32)))
        data_list_swapped.append(Data(x_l=x_i_r, adj_l=adj_i_r, mask_l=mask_r, x_r=x_i_l, adj_r=adj_i_l, mask_r=mask_l,
                                      y=torch.tensor(data[i][2]).to(torch.float32)))

    return data_list, data_list_swapped

# %%
# Generate and save datasets with different feature initializations
def main():
    generator1 = torch.Generator().manual_seed(42)

    # Dataset with random normal features
    data_rn, max_n_nodes_rn, max_min_ged_not_normalized_rn, max_diam_rn = generate_dataset(half_n_couples, features_init_rn)
    data_list_rn, data_list_swapped_rn = padding_and_masking_data(data_rn, max_n_nodes_rn)
    data_list_rn = data_list_rn + data_list_swapped_rn

    # Dataset with random uniform features
    data_ru, max_n_nodes_ru, max_min_ged_not_normalized_ru, max_diam_ru = generate_dataset(half_n_couples, features_init_ru)
    data_list_ru, data_list_swapped_ru = padding_and_masking_data(data_ru, max_n_nodes_ru)
    data_list_ru = data_list_ru + data_list_swapped_ru

    # Dataset with constant one features
    data_1, max_n_nodes_1, max_min_ged_not_normalized_1, max_diam_1 = generate_dataset(half_n_couples, features_init_1)
    data_list_1, data_list_swapped_1 = padding_and_masking_data(data_1, max_n_nodes_1)
    data_list_1 = data_list_1 + data_list_swapped_1

    # Save datasets to file
    torch.save(data_list_1, 'dataset_ones.pth')
    torch.save([max_n_nodes_1, max_min_ged_not_normalized_1, max_diam_1], 'dataset_ones_infos.pth')

    torch.save(data_list_rn, 'dataset_rn.pth')
    torch.save([max_n_nodes_rn, max_min_ged_not_normalized_rn, max_diam_rn], 'dataset_rn_infos.pth')

    torch.save(data_list_ru, 'dataset_ru.pth')
    torch.save([max_n_nodes_ru, max_min_ged_not_normalized_ru, max_diam_ru], 'dataset_ru_infos.pth')


# Run main only if executed as script
if __name__ == "__main__":
    main()