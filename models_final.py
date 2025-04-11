# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric
import torch.masked
from torch_geometric.data import Data, DenseDataLoader
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn.dense import dense_diff_pool, dense_mincut_pool, DenseGraphConv, DenseGINConv, DenseGCNConv
from torch_geometric.utils import to_torch_coo_tensor, to_dense_adj, to_edge_index

# Utility function to get activation function based on a string
def get_activation_function(activation: str) -> nn.Module:
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'linear':
        return nn.Identity()  # No activation
    else:
        raise NotImplementedError

# Utility function to create a GNN layer of a specific type
def get_gnn_type(channels: list, **kwargs) -> nn.Module:
    gnn_type = kwargs.get('gnn_type')
    if gnn_type == 'GCN':
        return DenseGCNConv(*channels)
    elif gnn_type == 'graph_conv':
        return DenseGraphConv(*channels)
    elif gnn_type == 'GIN':
        in_units, out_units = channels
        h_units = kwargs.get('h_')
        drop_out = kwargs.get('drop_out')
        act = kwargs.get('act')
        output_act = kwargs.get('output_act')
        return DenseGINConv(
            MLP(
                input_units=in_units,
                hidden_units=[h_units],
                output_units=out_units,
                drop_out=drop_out,
                act=act,
                output_act=output_act
            ),
            train_eps=True
        )
    elif gnn_type is None:
        raise Exception('Unknown GNN type')
    else:
        raise NotImplementedError

# MLP model used in the GIN and in the final classifier
class MLP(nn.Module):
    def __init__(self,
                 input_units: int,
                 hidden_units: list[int],
                 output_units: int = 1,
                 drop_out: float = 0,
                 act: str = "relu",
                 output_act: str = "leakyrelu",
                 alpha: float = 1.):
        super(MLP, self).__init__()
        self.name = "MLP"
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.output_act = output_act
        self.mlp_layers = nn.ModuleList()

        # Input layer
        self.mlp_layers.append(nn.Linear(input_units, hidden_units[0]))
        self.mlp_layers.append(get_activation_function(act))
        if drop_out > 0:
            self.mlp_layers.append(nn.Dropout(p=drop_out))

        # Hidden layers
        for i in range(len(hidden_units)):
            if i == len(hidden_units) - 1:
                self.mlp_layers.append(nn.Linear(hidden_units[i], output_units))
                break
            self.mlp_layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.mlp_layers.append(get_activation_function(act))
            if drop_out > 0:
                self.mlp_layers.append(nn.Dropout(p=drop_out))

        # Output activation layer
        self.output_layer = get_activation_function(output_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp_layers:
            x = layer(x)
        return self.alpha * self.output_layer(x)

# Main GNN architecture with multiple message-passing layers
class GNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 mp_layers: int = 3,
                 batch_norm: bool = False,
                 **kwargs):
        super(GNN, self).__init__()
        self.name = "GNN"
        self.gnn_type = "GCN" if kwargs.get('gnn_type') is None else kwargs.get('gnn_type')
        self.mp_layers = mp_layers
        self.batch_norm = batch_norm
        self.layers = torch.nn.ModuleList()

        if self.mp_layers == 0:
            raise NameError("GNN must have at least 1 message-passing layer")

        # Build the GNN layer stack
        for i in range(self.mp_layers):
            if i == 0:
                self.layers.append(get_gnn_type(channels=[in_channels, hidden_channels], **kwargs))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_channels))
            elif i == self.mp_layers - 1:
                self.layers.append(get_gnn_type(channels=[hidden_channels, out_channels], **kwargs))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(out_channels))
            else:
                self.layers.append(get_gnn_type(channels=[hidden_channels, hidden_channels], **kwargs))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, adj, mask=None):
        for layer in self.layers:
            if not isinstance(layer, nn.BatchNorm1d):
                x = F.relu(layer(x, adj, mask=mask))
            else:
                # Apply batch norm to the feature dimension
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x, adj

# Siamese network for graph similarity comparison
class SingleSiamese(nn.Module):
    def __init__(
            self,
            input_features_dim: int,
            state_dim: int,
            mp_layers: int,
            batch_norm: bool,
            global_pool_type: str,
            mlp_layers: list[int],
            mlp_act: str,
            mlp_dropout: float,
            mlp_alpha: float,
            **kwargs
    ):
        super().__init__()
        self.max_n_node = input_features_dim
        self.global_pool_type = global_pool_type

        # Shared GNN encoder for both branches
        self.gnn = GNN(in_channels=input_features_dim,
                       hidden_channels=state_dim,
                       out_channels=state_dim,
                       mp_layers=mp_layers,
                       batch_norm=batch_norm,
                       **kwargs)

        # Final MLP for output (e.g., similarity score)
        self.mlp = MLP(input_units=2 * state_dim,
                       hidden_units=mlp_layers,
                       act=mlp_act,
                       alpha=mlp_alpha,
                       drop_out=mlp_dropout)

    def forward(self, x_l, x_r, adj_l, adj_r, mask_l=None, mask_r=None):
        h_graph_representation_l = torch.tensor([], requires_grad=True)
        h_graph_representation_r = torch.tensor([], requires_grad=True)

        # Encode both graphs using the same GNN
        x_l, adj_l = self.gnn(x_l, adj_l, mask_l)
        x_r, adj_r = self.gnn(x_r, adj_r, mask_r)

        # Apply global pooling (mean, max, or sum)
        if self.global_pool_type == 'mean':
            h_graph_representation_l = torch.cat([h_graph_representation_l, x_l.mean(dim=1)], dim=1)
            h_graph_representation_r = torch.cat([h_graph_representation_r, x_r.mean(dim=1)], dim=1)
        elif self.global_pool_type == 'max':
            x_l_, _ = x_l.max(dim=1)
            x_r_, _ = x_r.max(dim=1)
            h_graph_representation_l = torch.cat([h_graph_representation_l, x_l_], dim=1)
            h_graph_representation_r = torch.cat([h_graph_representation_r, x_r_], dim=1)
        elif self.global_pool_type == 'sum':
            x_l_ = torch.sum(x_l, dim=1)
            x_r_ = torch.sum(x_r, dim=1)
            h_graph_representation_l = torch.cat([h_graph_representation_l, x_l_], dim=1)
            h_graph_representation_r = torch.cat([h_graph_representation_r, x_r_], dim=1)

        # Concatenate graph representations from both branches
        h_graph_representation = torch.cat([h_graph_representation_l, h_graph_representation_r], dim=1)

        # Predict output using the MLP (e.g., similarity score)
        out = self.mlp(h_graph_representation)

        return out

# Utility to save model parameters and optimizer state
def save_model(model, optimizer, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path + "/model_pars.pt")
