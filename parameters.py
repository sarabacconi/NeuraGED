# ====================== COMMON PARAMETERS ======================

num_features_ZINC = 27                   # Number of features per node in ZINC dataset
input_feature_dim = num_features_ZINC   # Input dimension for the GNN (node features)
state_dim = 128                          # Hidden state size used across GNN and MLP layers
batch_size = 512                         # Batch size for training
batch_norm = False                       # Whether to apply batch normalization in GNN layers
global_pool_type = 'sum'                # Pooling method: 'sum', 'mean', or 'max' across nodes

# MLP architecture: each element is the size of a layer
mlp_layers = [2 * state_dim, 10 * state_dim]  # Two layers: one intermediate (2×state_dim), one large (10×state_dim)
mlp_act = 'relu'                        # Activation function used in MLP layers
mlp_dropout = 0.                        # Dropout rate for MLP (0 means no dropout)
mlp_alpha = 1                           # Alpha for activation functions like LeakyReLU or SELU (if used)

n_epochs = 5                         # Maximum number of training epochs
lr = 0.000001                           # Learning rate for the optimizer

gnn_type: str = 'GIN'                   # Type of Graph Neural Network (e.g., GCN, GAT, GIN)
mp_layers: int = 3                      # Number of Message Passing layers in the GNN

# ====================== GIN-SPECIFIC PARAMETERS ======================

gin_hidden: int = 2 * state_dim         # Hidden dimension of each GIN layer
gin_drop_out: float = 0.                # Dropout rate in GIN layers (0 = no dropout)
gin_act: str = 'selu'                   # Activation function used inside GIN layers
gin_out_act: str = 'selu'               # Activation function used in the final GIN layer output

# ====================== DISTANCE FUNCTION PARAMETER ======================

distance = 'mlp'                        # How to compute similarity between graph embeddings:
                                       # 'l2' = L2 distance, 'mlp' = pass concatenated embeddings through MLP

patience = 10
