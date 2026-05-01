# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrysCo is a hybrid graph-transformer neural network for predicting inorganic material properties. The model combines Edge Graph Attention Networks (EGAT), Transformer architectures, and SE(3)-equivariant layers to process crystal structures and predict material properties from the Materials Project database.

## Core Architecture

### Model Components
- **CrysCo**: Main model class combining EGAT layers, transformers, and residual networks
- **EGAT**: Edge Graph Attention Networks for processing atomic graphs with edge features
- **Transformer**: Standard transformer architecture with positional encoding for sequences
- **SE**: SE(3)-equivariant layers for handling 3D geometric information
- **MLP**: Multi-layer perceptron utilities for various network components

### Data Pipeline
The training pipeline processes crystal structures through several stages:

1. **CIF Processing**: Crystal structures are loaded from CIF files using ASE
2. **Graph Construction**: Atomic structures converted to PyTorch Geometric graphs with:
   - Node features: atomic properties, positions
   - Edge features: bond distances, angles, dihedral angles
3. **Feature Engineering**: Additional structural features extracted via `extracted_features.py`
4. **Data Loading**: Custom PyTorch Geometric dataset classes handle batching and transforms

### Key Data Flow
```
CIF Files → Graph Construction → Feature Extraction → Model Training → Property Prediction
```

## Development Commands

### Data Preparation
```bash
# Generate training data from CIF files and CSV property data
python data_preparation.py --path_to_cif_structure <cif_dir> --first_csv_file <formulas.csv> --second_csv_file <formulas_with_ids.csv> --output <output_dir>
```

### Model Training
```bash
# Train the CrysCo model
python main.py --data_dir <project_root> --data <processed_data.pt>
```

### Prediction
Use `prediction.ipynb` notebook for inference with trained models.

## Configuration

Model parameters are defined in `main.py`:
- **Architecture**: `out_dims`, `d_model`, `N` (transformer layers), `heads`, `numb_EGAT`
- **Training**: `epochs`, `lr`, `batch_size`, `optimizer`, `scheduler`
- **Target**: `target_index` selects which material property to predict

## File Structure Notes

### Core Files
- `CrysCo.py`: Main model architecture
- `main.py`: Training script and parameter configuration
- `utils_train.py`: Training loop implementations
- `data.py`: Dataset classes and data loading utilities
- `utils.py`: General utilities for data processing and transformations

### Data Processing
- `data_preparation.py`: Primary data preprocessing pipeline
- `data_prep.py`: Alternative/older data preprocessing (check if redundant)
- `graph_dihedral.py`: Graph construction with geometric features
- `extracted_features.py`: Handcrafted feature extraction

### Model Components
- `EGAT.py`: Edge Graph Attention Network implementations
- `transformer.py`: Transformer architecture components
- `SE.py`: SE(3)-equivariant layers
- `MLP.py`: Multi-layer perceptron utilities

### Data Directories
- `material/`: Sample CIF files and CSV data
- `processed/`: Generated PyTorch datasets (.pt files)

## Important Development Notes

- The model uses PyTorch Geometric for graph neural network operations
- Training supports both single-GPU and distributed training
- Graph construction includes 4-body interactions via dihedral angles
- Model checkpoints are saved with `.pth` extension
- All random seeds are set to 42 for reproducibility

## Dependencies

Key packages required:
- PyTorch & PyTorch Geometric
- ASE (Atomic Simulation Environment)
- NumPy, Pandas, SciPy
- scikit-learn for preprocessing

Note: No `requirements.txt` currently exists - this should be created for proper dependency management.