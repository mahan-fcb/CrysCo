# CrysCo: Hybrid Graph-Transformer for Materials Property Prediction

A PyTorch implementation of a hybrid graph-transformer neural network that combines Edge Graph Attention Networks (EGAT) and Transformer architectures for predicting inorganic material properties.

**Paper**: "Accelerating materials property prediction via a hybrid Transformer Graph framework that leverages four body interactions"

## Features

- **Hybrid Architecture**: Combines EGAT, Transformer, and SE(3)-equivariant layers
- **4-body Interactions**: Incorporates dihedral angles for enhanced geometric understanding  
- **Materials Project Integration**: Trained on MP21 datasets for 8 material properties
- **Scalable Training**: Supports single-GPU and distributed training

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (recommended for GPU acceleration)

### Install from source
```bash
git clone https://github.com/user/CrysCo.git
cd CrysCo
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Data Preparation
Download CIF files from Materials Project and prepare the dataset:

```bash
python data_preparation.py \
    --path_to_cif_structure path/to/cif/files \
    --first_csv_file formulas.csv \
    --second_csv_file formulas_with_ids.csv \
    --output processed/
```

**Note**: The first CSV should contain regular pretty formulas, while the second should include formulas with unique identifiers for materials with similar compositions but different structures.

### 2. Model Training
```bash
python main.py --data_dir . --data processed/dataset.pt
```

### 3. Prediction
Use the `prediction.ipynb` notebook for inference with trained models.

## Dataset

We primarily use the **Materials Project MP21** dataset covering 8 properties:
- Formation energy
- Band gap
- Bulk modulus
- Shear modulus
- And 4 additional properties

The dataset includes diverse inorganic crystal structures with corresponding calculated properties.

## Model Architecture

### Core Components
- **EGAT Layers**: Edge Graph Attention Networks for processing atomic graphs
- **Transformer**: Standard transformer with positional encoding
- **SE(3) Layers**: Equivariant layers for 3D geometric information
- **Residual Networks**: Skip connections for improved gradient flow

### Data Flow
```
CIF Files → Graph Construction → Feature Extraction → Model Training → Property Prediction
```

## Configuration

Key parameters in `main.py`:

```python
model_parameters = {
    "out_dims": 64,
    "d_model": 128,
    "N": 3,              # Transformer layers
    "heads": 4,          # Attention heads
    "numb_EGAT": 5,      # Number of EGAT layers
    "epochs": 800,
    "lr": 0.006,
    "batch_size": 80
}
```

## Examples

Training with custom data:
```bash
# Example with absolute paths
python data_preparation.py \
    "/path/to/materials/cif_files" \
    "properties.csv" \
    "properties_with_ids.csv" \
    "/path/to/output/dataset.pt"

python main.py --data_dir "/path/to/project" --data "dataset.pt"
```

## Project Structure

```
CrysCo/
├── crysco/                    # Main Python package
│   ├── models/               # Neural network models
│   │   ├── CrysCo.py        # Main hybrid model
│   │   ├── EGAT.py          # Edge Graph Attention Networks
│   │   ├── transformer.py   # Transformer components
│   │   ├── SE.py            # SE(3)-equivariant layers
│   │   └── MLP.py           # Multi-layer perceptrons
│   ├── data/                # Data loading utilities
│   │   └── data.py          # Dataset classes and loaders  
│   └── utils/               # Training utilities
│       └── utils_train.py   # Training loops and model setup
├── scripts/                  # Standalone scripts
│   └── preprocessing/       # Data preprocessing scripts
│       ├── data_preparation.py
│       ├── graph_dihedral.py
│       └── extracted_features.py
├── tests/                   # Test suite
├── docs/                   # Documentation
├── main.py                 # Main training script
├── material/              # Sample data
├── processed/            # Generated datasets
└── prediction.ipynb     # Inference notebook
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