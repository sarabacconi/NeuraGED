# %%
import torch
import os
from models_final import *
import json
import numpy as np
from parameters import *
from torch_geometric.data import DenseDataLoader

# %%
# Setup for cross-validation and dataset loading
results = {}                # Dictionary to store metrics for each fold
averaged_results = {}       # Dictionary to store average metrics across all folds
model_name = "params_GIN_5" # Identifier for the model being trained


# Dataset selection (choose one among the options below)
dataset_name = os.path.join("NeuraGED_github", "dataset_ones.pth")
dataset_infos_name = os.path.join("NeuraGED_github", "dataset_ones_infos.pth")

# Load dataset and metadata (e.g., max number of nodes, GED range, max diameter)
dataset = torch.load(dataset_name)
dataset_infos = torch.load(dataset_infos_name)
max_diam = dataset_infos[2]   # Max graph diameter used to set number of GNN layers
mp_layers = max_diam          # Dynamically adjust number of message passing layers

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Early stopping utility to stop training when validation loss stops improving
class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience            # Number of epochs to wait before stopping
        self.min_delta = min_delta          # Minimum improvement to reset the counter
        self.counter = 0                    # Tracks how many epochs since last improvement
        self.min_validation_loss = float('inf')  # Lowest validation loss seen
        self.train_loss = None              # Best training loss seen
        self.best_model_params = None       # State dict of the best model

    def early_stop(self, validation_loss, train_loss, model_state_dict):
        if validation_loss < self.min_validation_loss:
            # Update best values and reset counter
            self.min_validation_loss = validation_loss
            self.train_loss = train_loss
            self.best_model_params = model_state_dict
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # Increment counter if no improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Trigger early stop
        return False


# %%
# Initialize the GNN-based Siamese model and optimizer
model = SingleSiamese(
    input_features_dim=input_feature_dim,
    state_dim=state_dim,
    mp_layers=mp_layers,
    batch_norm=batch_norm,
    global_pool_type=global_pool_type,
    mlp_layers=mlp_layers,
    mlp_act=mlp_act,
    mlp_dropout=mlp_dropout,
    mlp_alpha=mlp_alpha,
    gnn_type=gnn_type,
    h_=gin_hidden,
    drop_out=gin_drop_out,
    output_act=gin_out_act,
    act=gin_act
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
earlystopper = EarlyStopper(patience=patience)

generator1 = torch.Generator().manual_seed(42)  # Reproducibility



# Split train + val  + test(70/15/15)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15], generator=generator1)

# Create data loaders
train_loader = DenseDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DenseDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DenseDataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# %%
# TRAINING FUNCTION: trains the model on one epoch over the dataset
def train(dataset):
    model.train()         # Set model to training mode
    loss_all = 0.0        # Accumulate total loss for averaging

    for D in dataset:     # Iterate over batches
        D = D.to(device)
        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        output = model(
            D.x_l, D.x_r,
            D.adj_l, D.adj_r,
            D.mask_l, D.mask_r
        )

        # Compute loss (Mean Squared Error between predicted and true GED)
        loss_mse = F.mse_loss(output, D.y.unsqueeze(1))
        loss = loss_mse

        # Backward pass and optimizer step
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

    return loss_all / len(dataset)  # Return average loss over all batches


# %%
# EVALUATION FUNCTION: evaluates the model on validation or test data
@torch.no_grad()
def test(loader):
    model.eval()  # Set model to evaluation mode (no dropout, etc.)
    loss_per_batch = []

    for D in loader:
        D = D.to(device)

        # Forward pass
        pred = model(
            D.x_l, D.x_r,
            D.adj_l, D.adj_r,
            D.mask_l, D.mask_r
        )

        # Compute MSE loss
        loss = F.mse_loss(pred, D.y.unsqueeze(1))
        loss_per_batch.append(loss.item())

    return np.sum(loss_per_batch) / len(loader)  # Average loss across all batches

# Training loop with early stopping
for epoch in range(1, n_epochs + 1):
    train_loss = train(train_loader)
    val_loss = test(val_loader)
    print(f'Epoch: {epoch:03d}, Train MSE: {train_loss:.5f}, Val MSE: {val_loss:.5f}')

    if earlystopper.early_stop(val_loss, train_loss, model.state_dict()):
        print(f"\nEarly stopping at epoch: {epoch}, best_val_mse: {earlystopper.min_validation_loss}")
        model.load_state_dict(earlystopper.best_model_params)
        break

# Final test evaluation for the current fold
test_loss = test(test_loader)
print(f'Test MSE: {test_loss:.5f}')
