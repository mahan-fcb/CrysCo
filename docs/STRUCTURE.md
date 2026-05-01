# Repository Structure

This document describes the organization of the CrysCo repository.

## Directory Structure

```
CrysCo/
├── crysco/                          # Main Python package
│   ├── models/                      # Neural network models
│   │   ├── CrysCo.py                # Main hybrid model
│   │   ├── EGAT.py                  # Edge Graph Attention Networks
│   │   ├── MLP.py                   # Multi-layer perceptrons
│   │   ├── SE.py                    # SE(3)-equivariant layers
│   │   └── transformer.py           # Transformer components
│   ├── data/                        # Data loading utilities
│   │   └── data.py                  # Dataset classes and loaders
│   └── utils/                       # Training utilities
│       ├── utils_train.py           # Training loops and model setup
│       └── utils.py                 # General utilities
├── scripts/                         # Standalone scripts
│   └── preprocessing/               # Data preprocessing
│       ├── data_preparation.py      # CIF → .pt pipeline
│       ├── graph_dihedral.py        # Graph class with angles/dihedrals
│       ├── extracted_features.py    # Handcrafted feature extraction
│       └── get_MP.py                # Materials Project downloader
├── material/                        # Sample CIF files and CSV data
├── tests/                           # Test suite
├── docs/                            # Documentation
│   └── STRUCTURE.md                 # This file
├── main.py                          # Main training script (CLI)
├── prediction.ipynb                 # Inference notebook
├── dictionary_default.json          # Per-element feature dictionary
├── mat2vec.csv                      # Element embeddings
├── requirements.txt                 # Dependencies
└── setup.py                         # Package installation
```

## Key Files

### Core Training
- `main.py` - CLI training script with full argparse support
- `crysco/models/CrysCo.py` - Main model architecture
- `crysco/utils/utils_train.py` - Training loop implementations

### Data Processing
- `crysco/data/data.py` - Data loading, splitting, `drop_last=True` on train loader
- `scripts/preprocessing/data_preparation.py` - Full CIF → .pt pipeline
- `scripts/preprocessing/graph_dihedral.py` - `Graph` class generating 210-dim angle features

### Configuration / Data
- `dictionary_default.json` - 114-dim per-element feature dictionary
- `mat2vec.csv` - Element embeddings for transformer encoder
- `material/` - Sample CIF files and CSV property data for testing

## Import Structure

Imports follow Python package conventions:

```python
# From main.py
from crysco.models.CrysCo import CrysCo
from crysco.utils.utils_train import train_model, train_one_epoch, model_setup, evaluate
from crysco.data.data import setup_data_loaders, get_dataset, StructureDataset, GetY

# Within the crysco package
from .MLP import MLP
from ..utils.utils import EDM_CsvLoader
```

## Usage Summary

```bash
# Preprocessing: CIF → .pt
python scripts/preprocessing/data_preparation.py \
    <cif_dir>/ <formulas_with_ids.csv> <formulas.csv> <output_dir>/processed/<out>.pt

# Training
python main.py --data_dir <dir> --data <file>.pt --device cuda --epochs 100

# Install as package
pip install -e .
```

See the top-level `README.md` for full usage examples with all CLI flags.
