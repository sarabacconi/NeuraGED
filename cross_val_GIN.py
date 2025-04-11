# %%
import torch
import os
from models_final import *
import json
import numpy as np
from parameters import *
from torch_geometric.data import DenseDataLoader

# %%
# Define the number of folds, batch_size and load dataset
results = {}
averaged_results = {}
model_name = "params_GIN_5"
k_folds = 5  # Number of folds for cross-validation

# Load the dataset and metadata
dataset_name = os.path.join("NeuraGED_github", "dataset_ones.pth")
dataset_infos_name = os.path.join("NeuraGED_github", "dataset_ones_infos.pth")
dataset = torch.load(dataset_name)  # Load preprocessed dataset
dataset_infos = torch.load(dataset_infos_name)
max_diam = dataset_infos[2]  # Typically used for message passing layers
mp_layers = max_diam  # Number of GNN layers

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
#print(n_epochs)

# %%
# %%
class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience  # Number of epochs to wait without improvement
        self.min_delta = min_delta  # Minimum change to qualify as an improvement
        self.counter = 0  # Counter for non-improving epochs
        self.min_validation_loss = float('inf')
        self.train_loss = None
        self.best_model_params = None  # Store best model parameters

    def early_stop(self, validation_loss, train_loss, model_state_dict):
        # Check if validation loss improved
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.train_loss = train_loss
            self.best_model_params = model_state_dict
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# %%


# %%
# Initialize the model and optimizer
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



# %%
# TRAINING FUNCTION
def train(dataset):
    model.train()
    loss_all = 0
    for D in dataset:
        D = D.to(device)
        optimizer.zero_grad()
        output = model(D.x_l, D.x_r, D.adj_l, D.adj_r, D.mask_l, D.mask_r)
        loss_mse = F.mse_loss(output, D.y.unsqueeze(1))  # Mean squared error loss
        loss = loss_mse
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(dataset)


# %%
# TEST FUNCTION
@torch.no_grad()
def test(loader):
    model.eval()
    loss_per_batch = []
    for D in loader:
        D = D.to(device)
        pred = model(D.x_l, D.x_r, D.adj_l, D.adj_r, D.mask_l, D.mask_r)
        loss = F.mse_loss(pred, D.y.unsqueeze(1))
        loss_per_batch.append(loss.item())
    return np.sum(loss_per_batch) / len(loader)


# %%
def split_in_k_folds(k_folds, dataset):
    l = len(dataset)
    elements_x_fold = int(l / k_folds)
    folds_list = []
    for k in range(k_folds):
        if k == 0:
            fold = dataset[:elements_x_fold]
        elif k != k_folds - 1:
            start = k * elements_x_fold
            stop = start + elements_x_fold
            fold = dataset[start:stop]
        else:
            fold = dataset[-elements_x_fold:]  # Last fold
        folds_list.append(fold)
    return folds_list



folds_list = split_in_k_folds(k_folds, dataset)
# print(len(folds_list[0]), len(folds_list[1]), len(folds_list[2]), len(folds_list[3]), len(folds_list[4]))
# print(len(folds_list[0]) + len(folds_list[1]) + len(folds_list[2]) + len(folds_list[3]) + len(folds_list[4]) == len(
#     dataset))


# %%
def reset_weights(m):
    # Recursively reset model weights
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()



# %%
# Loop through each fold
generator1 = torch.Generator().manual_seed(42)  # Reproducibility

for fold in range(k_folds):
    print(f"Fold {fold + 1}\n-------")

    # Reset model and optimizer
    model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopper()

    # Prepare train/validation/test datasets
    folds_list = split_in_k_folds(k_folds, dataset)
    test_dataset = folds_list[fold]
    folds_list_ = folds_list.pop(fold)
    train_and_val_dataset = folds_list_

    # Split train + val (85/15)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_and_val_dataset, [0.85, 0.15], generator=generator1)

    # Create data loaders
    train_loader = DenseDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DenseDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DenseDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop with early stopping
    for epoch in range(1, n_epochs + 1):
        train_loss = train(train_loader)
        val_loss = test(val_loader)
        print(f'Epoch: {epoch:03d}, Train MSE: {train_loss:.5f}, Val MSE: {val_loss:.5f}')

        if early_stopper.early_stop(val_loss, train_loss, model.state_dict()):
            print(f"\nEarly stopping at epoch: {epoch}, best_val_mse: {early_stopper.min_validation_loss}")
            model.load_state_dict(early_stopper.best_model_params)
            break

    # Final test evaluation for the current fold
    test_loss = test(test_loader)
    print(f'Test MSE: {test_loss:.5f}')

    results[fold] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss
    }

    # Save results for each fold
    with open("cross_validation_results_params_GIN_1.txt", "w") as file:
        json.dump(results, file, indent=4)

# %%
# Display and save averaged performance across folds
print(f'K-FOLD CROSS VALIDATION AVERAGED RESULTS FOR {k_folds} FOLDS AND MODEL {model_name}')
print('--------------------------------')

sum_train_loss, sum_val_loss, sum_test_loss = 0, 0, 0
for key in results:
    sum_train_loss += results[key]["train_loss"]
    sum_val_loss += results[key]["val_loss"]
    sum_test_loss += results[key]["test_loss"]

average_train_loss = sum_train_loss / k_folds
average_val_loss = sum_val_loss / k_folds
average_test_loss = sum_test_loss / k_folds

averaged_results[model_name] = {
    'average_train_loss': average_train_loss,
    'average_val_loss': average_val_loss,
    'average_test_loss': average_test_loss
}

# Save final averaged results to JSON file
with open("cross_validation_averaged_results.txt", "w") as file:
    json.dump(averaged_results, file, indent=4)

print(f'average_train_loss : {average_train_loss}, average_val_loss: {average_val_loss}, average_test_loss:{average_test_loss}')
