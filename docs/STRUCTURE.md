# Repository Structure

This document describes the organization of the CrysCo repository after professional restructuring.

## Directory Structure

```
CrysCo/
├── crysco/                    # Main Python package
│   ├── models/               # Neural network models
│   │   ├── CrysCo.py        # Main hybrid model
│   │   ├── EGAT.py          # Edge Graph Attention Networks
│   │   ├── MLP.py           # Multi-layer perceptrons
│   │   ├── SE.py            # SE(3)-equivariant layers
│   │   └── transformer.py   # Transformer components
│   ├── data/                # Data loading utilities
│   │   └── data.py          # Dataset classes and loaders
│   └── utils/               # Training utilities
│       ├── utils_train.py   # Training loops and setup
│       └── utils.py         # General utilities
├── scripts/                  # Standalone scripts
│   ├── preprocessing/       # Data preprocessing scripts
│   │   ├── data_preparation.py
│   │   ├── graph_dihedral.py
│   │   ├── extracted_features.py
│   │   └── get_MP.py
│   ├── training/           # Training scripts (future)
│   └── testing/            # Testing scripts (future)
├── tests/                   # Test suite
│   ├── simple_test.py      # Basic functionality tests
│   └── test_imports.py     # Import verification tests
├── docs/                   # Documentation
├── examples/              # Usage examples (future)
├── material/             # Sample data
├── processed/           # Processed datasets
├── main.py             # Main training script
└── requirements.txt    # Dependencies
```

## Key Files

### Core Training
- `main.py` - Main training script, imports from `crysco` package
- `crysco/models/CrysCo.py` - Main model architecture
- `crysco/utils/utils_train.py` - Training loop implementations

### Data Processing  
- `crysco/data/data.py` - Data loading and splitting utilities
- `scripts/preprocessing/` - Preprocessing scripts for raw data

### Testing
- `tests/` - All test files for verifying functionality

## Import Structure

After reorganization, imports follow Python package conventions:

```python
# Main training script
from crysco.models.CrysCo import CrysCo
from crysco.utils.utils_train import train_model
from crysco.data.data import setup_data_loaders

# Within package modules
from ..models.CrysCo import CrysCo  # Relative import
from .MLP import MLP                # Same directory
```

## Benefits of New Structure

1. **Professional Organization** - Clear separation of concerns
2. **Maintainable** - Easy to find and modify specific components
3. **Testable** - Isolated test suite in dedicated directory
4. **Extensible** - Easy to add new models, utilities, or scripts
5. **Installable** - Can be installed as Python package via `pip install -e .`

## Usage

```bash
# Training (from repository root)
python main.py --data_dir . --data M.pt

# Preprocessing (standalone scripts)
python scripts/preprocessing/data_preparation.py --help

# Testing
python -m pytest tests/

# Install as package
pip install -e .
```