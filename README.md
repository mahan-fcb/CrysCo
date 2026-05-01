# CrysCo: Hybrid Graph-Transformer for Materials Property Prediction

A PyTorch implementation of a hybrid graph-transformer neural network that combines Edge Graph Attention Networks (EGAT) and Transformer architectures for predicting inorganic material properties.

**Paper**: "Accelerating materials property prediction via a hybrid Transformer Graph framework that leverages four body interactions"

## Features

- **Hybrid Architecture**: Combines EGAT, Transformer, and SE(3)-equivariant layers
- **4-body Interactions**: Incorporates bond angles and dihedral angles for enhanced geometric understanding
- **Materials Project Integration**: Trained on MP21 datasets for 8 material properties
- **Flexible Training**: Single-GPU, multi-GPU, and CPU support via CLI arguments

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA (recommended for GPU acceleration)

### Install from source
```bash
git clone https://github.com/user/CrysCo.git
cd CrysCo
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
CrysCo/
├── crysco/                          # Main Python package
│   ├── models/                      # Neural network models
│   │   ├── CrysCo.py                # Main hybrid model
│   │   ├── EGAT.py                  # Edge Graph Attention Networks
│   │   ├── transformer.py           # Transformer components
│   │   ├── SE.py                    # SE(3)-equivariant layers
│   │   └── MLP.py                   # Multi-layer perceptrons
│   ├── data/                        # Data loading utilities
│   │   └── data.py                  # Dataset classes and loaders
│   └── utils/                       # Training utilities
│       ├── utils.py                 # General utilities
│       └── utils_train.py           # Training loops and model setup
├── scripts/                         # Standalone scripts
│   └── preprocessing/               # Data preprocessing
│       ├── data_preparation.py      # CIF → .pt pipeline
│       ├── graph_dihedral.py        # Graph construction with angles
│       └── extracted_features.py    # Handcrafted feature extraction
├── material/                        # Sample CIF files and CSV data
├── tests/                           # Test suite
├── docs/                            # Documentation
├── main.py                          # Main training script (CLI)
├── prediction.ipynb                 # Inference notebook
├── dictionary_default.json          # Atom feature dictionary
├── mat2vec.csv                      # Element embeddings
├── requirements.txt                 # Dependencies
└── setup.py                         # Package installation
```

## Quick Start

### 1. Data Preprocessing

Convert CIF files and CSV property data into a PyTorch Geometric dataset:

```bash
python scripts/preprocessing/data_preparation.py \
    <cif_directory> \
    <formulas_with_ids.csv> \
    <formulas.csv> \
    <output_dir>/processed/<output>.pt
```

**Arguments:**
- `cif_directory`: Directory containing the `.cif` structure files
- `formulas_with_ids.csv`: CSV with formulas including unique IDs (compositional data)
- `formulas.csv`: CSV with formulas and target property values
- `output`: Path to output `.pt` file (must be inside a `processed/` subdirectory)

**Example with sample data:**
```bash
python scripts/preprocessing/data_preparation.py \
    /home/user/CrysCo/material/ \
    /home/user/CrysCo/material/ehs.csv \
    /home/user/CrysCo/material/eh.csv \
    /home/user/CrysCo/material/processed/material_data.pt
```

The preprocessing pipeline:
1. Reads CIF structures with ASE
2. Builds atomic graphs with distance-based edges
3. Computes bond angles and dihedral angles (4-body interactions) via the `Graph` class
4. Extracts per-atom features from `dictionary_default.json`
5. Computes per-structure human features (density, symmetry, etc.) via matminer
6. Saves as a PyTorch Geometric `InMemoryDataset`

### 2. Model Training

Train CrysCo with full CLI control over hyperparameters:

```bash
python main.py \
    --data_dir <dataset_directory> \
    --data <dataset_file.pt> \
    --device cuda \
    --epochs 100 \
    --batch_size 80 \
    --lr 0.006 \
    --train_ratio 0.85 \
    --val_ratio 0.05 \
    --test_ratio 0.10 \
    --model_path my_model.pth \
    --job_name my_job
```

**CLI Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | required | Directory containing the dataset (with `processed/` subfolder) |
| `--data` | str | required | Dataset file name (e.g., `material_data.pt`) |
| `--device` | str | `cuda` | Device: `cuda` or `cpu` (falls back to CPU if CUDA unavailable) |
| `--epochs` | int | 100 | Number of training epochs |
| `--batch_size` | int | 80 | Batch size (use `drop_last=True` internally) |
| `--lr` | float | 0.006 | Learning rate |
| `--train_ratio` | float | 0.85 | Training data fraction |
| `--val_ratio` | float | 0.05 | Validation data fraction |
| `--test_ratio` | float | 0.10 | Test data fraction |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--model_path` | str | `my_model_shear2.pth` | Path to save trained model |
| `--job_name` | str | `my_train_job` | Prefix for output CSV files |

**Quick test with sample data (3 epochs):**
```bash
python main.py \
    --data_dir /home/user/CrysCo/material/ \
    --data material_data.pt \
    --device cuda \
    --epochs 3 \
    --batch_size 3 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --test_ratio 0.2 \
    --model_path material_model.pth \
    --job_name material_test
```

**Full training on GPU:**
```bash
python main.py \
    --data_dir ./my_dataset/ \
    --data dataset.pt \
    --device cuda \
    --epochs 800 \
    --batch_size 80
```

**CPU training (for debugging):**
```bash
python main.py \
    --data_dir ./my_dataset/ \
    --data dataset.pt \
    --device cpu \
    --epochs 3 \
    --batch_size 4
```

### 3. Testing / Evaluation

After training, evaluation metrics are automatically computed on train/val/test splits and printed:

```
Train Error: 0.01234
Val Error:   0.05678
Test Error:  0.06789
```

Predictions for each split are saved as CSV files (named with the `--job_name` prefix):
- `<job_name>_train_outputs.csv`
- `<job_name>_val_outputs.csv`
- `<job_name>_test_outputs.csv`

These files contain the ground-truth targets and model predictions for each sample, enabling further analysis.

### 4. Prediction on New Data

Use the `prediction.ipynb` notebook for inference on new crystal structures with a trained model.

Workflow:
1. Preprocess new CIF files into a `.pt` dataset (same as training)
2. Load the trained model from `<model_path>.pth`
3. Run predictions via the notebook's inference cell

## Dataset

Primary dataset: **Materials Project MP21** covering 8 properties:
- Formation energy
- Band gap
- Bulk modulus
- Shear modulus
- And 4 additional properties

The dataset includes diverse inorganic crystal structures with corresponding calculated properties.

## Model Architecture

### Core Components
- **EGAT Layers**: Edge Graph Attention Networks for processing atomic graphs with edge features
- **Transformer**: Standard transformer with compositional positional encoding
- **SE(3) Layers**: Equivariant layers for 3D geometric information
- **Residual Networks**: Skip connections for improved gradient flow

### Input Features per Atom
- **Atomic features** (114-dim): From `dictionary_default.json`
- **Angle features** (210-dim): 144 angle cosines + 66 dihedral angles from the `Graph` class
- **Edge features** (50-dim): Gaussian-smeared bond distances

### Data Flow
```
CIF Files → Graph Construction → Angle/Dihedral Features → Model Training → Property Prediction
```

## Configuration

Default model hyperparameters (in `main.py`):

```python
model_parameters = {
    "out_dims": 64,
    "d_model": 128,
    "N": 3,                # Transformer layers
    "heads": 4,            # Attention heads
    "numb_EGAT": 5,        # Number of EGAT layers
    "numb_GATGCN": 1,
    "epochs": 100,
    "lr": 0.006,
    "batch_size": 80,
    "optimizer": "AdamW",
    "scheduler": "ReduceLROnPlateau",
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Citation

If you use this code, please cite our manuscript and the following papers:

1. Omee, Sadman Sadeed, et al. "Scalable deeper graph neural networks for high-performance materials property prediction." *Patterns* 3.5 (2022).

2. Wang, Anthony Yu-Tung, et al. "Compositionally restricted attention-based network for materials property predictions." *npj Computational Materials* 7.1 (2021): 77.

## Contact

For questions or issues, please contact: mohammad73madani73@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
