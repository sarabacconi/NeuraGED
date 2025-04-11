# NeuraGED
Implementation of NeuraGED model


This repository provides an implementation of a Siamese Graph Neural Network (GNN) architecture for graph similarity estimation using PyTorch Geometric. The system includes tools to generate synthetic graph datasets, train and evaluate GNN-based models, and perform K-Fold cross-validation with early stopping.

---

## 📁 Repository Structure

```
.
├── dataset_generator.py     # Script to generate and preprocess synthetic graph datasets
├── parameters.py            # Global configuration and model/training hyperparameters
├── models_final.py          # Definition of the Siamese GNN model
├── training.py              # Training loop (single-run mode)
├── cross_val_GIN.py         # K-Fold cross-validation with early stopping
├── README.md                # Project documentation
```

---

## 📦 Dependencies

- Python ≥ 3.8
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- Matplotlib

Install with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install networkx matplotlib
```

---

## 📊 Dataset Generation

To create a synthetic dataset of graph pairs:

```bash
python dataset_generator.py
```

This script generates three datasets with node features initialized using:
- Random normal distribution
- Random uniform distribution
- All-ones

Each pair consists of a base graph and a modified version with added nodes/edges. GED (Graph Edit Distance) is approximated and used as the target regression value.

The datasets are saved as:

```
dataset_ones.pth
dataset_rn.pth
dataset_ru.pth
```

And their associated info files:

```
dataset_ones_infos.pth
dataset_rn_infos.pth
dataset_ru_infos.pth
```

---

## 🧠 Model Architecture

The model is a **Siamese GNN**, where each subnetwork is a stack of GIN (Graph Isomorphism Network) layers. Node embeddings are pooled into graph embeddings using a global pooling strategy (sum/mean/max), and a final MLP estimates the distance between the embeddings of two input graphs.

Model and training parameters are defined in `parameters.py`.

---

## 🏋️ Training

To train the model on the dataset:

```bash
python training.py
```

> Make sure to modify `training.py` to load your preferred dataset (e.g., `dataset_rn.pth`).

---

## 🔁 Cross-Validation

To run 5-fold cross-validation with early stopping:

```bash
python cross_val_GIN.py
```

Results will be saved in:
- `cross_validation_results_*.txt` — per-fold results
- `cross_validation_averaged_results.txt` — average results across folds

---

## ⚙️ Configuration

All important parameters such as:
- Feature dimensions
- Learning rate
- Number of message passing layers
- GNN/MLP architecture
- Batch size and number of epochs

...are defined in `parameters.py` for easy modification and reuse.

---

## 📈 Output

- Training, validation, and test MSE loss
- Estimated vs true GED
- Model saved in memory (or optionally to disk)
- Optional: graph visualization for debugging (see `draw_graph()` in `dataset_generator.py`)

---

## 📌 To Do

- [ ] Add visualization of predictions vs targets
- [ ] Add support for classification (e.g., similar vs dissimilar)
- [ ] Enable model checkpoint saving/loading
- [ ] Add CLI interface for config selection

---

## 📄 License

MIT License. Feel free to reuse and modify this code for your own research or application.

---

## ✉️ Contact

For questions or collaborations, feel free to open an issue or contact me via GitHub.

