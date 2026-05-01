# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrysCo is a hybrid graph-transformer neural network for predicting inorganic material properties. The model combines Edge Graph Attention Networks (EGAT), Transformer architectures, and SE(3)-equivariant layers to process crystal structures and predict material properties from the Materials Project database.

## Core Architecture

### Model Components
- **CrysCo**: Main model class combining EGAT layers, transformers, and residual networks
- **EGAT**: Edge Graph Attention Networks for processing atomic graphs with edge features
- **Transformer**: Standard transformer architecture with compositional positional encoding
- **SE**: SE(3)-equivariant layers for handling 3D geometric information
- **MLP**: Multi-layer perceptron utilities for various network components

### Data Pipeline
The preprocessing pipeline processes crystal structures through these stages:

1. **CIF Processing**: Crystal structures are loaded from CIF files using ASE
2. **Graph Construction**: Atomic structures converted to PyTorch Geometric graphs with:
   - Node features: 114-dim atomic features from `dictionary_default.json`
   - Edge features: 50-dim Gaussian-smeared bond distances
   - Angle features: 210-dim (144 angle cosines + 66 dihedral angles) via the `Graph` class in `graph_dihedral.py`
3. **Human Features**: Per-structure features (density, symmetry) via matminer / `extracted_features.py`
4. **Data Loading**: Custom PyTorch Geometric `InMemoryDataset` handles batching and transforms

### Key Data Flow
```
CIF Files → Graph Construction → Angle/Dihedral Features → Model Training → Property Prediction
```

## Development Commands

### Data Preparation
```bash
python scripts/preprocessing/data_preparation.py \
    <cif_directory> \
    <formulas_with_ids.csv> \
    <formulas.csv> \
    <output_dir>/processed/<output>.pt
```

The output `.pt` file MUST be inside a `processed/` subdirectory — the `get_dataset` loader expects this layout.

### Model Training
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

All hyperparameters are exposed via CLI flags; see `main.py` for full argument list.

### Prediction
Use `prediction.ipynb` notebook for inference with trained models.

## Configuration

Default model parameters are defined in `main.py` (`model_parameters` dict):
- **Architecture**: `out_dims=64`, `d_model=128`, `N=3` (transformer layers), `heads=4`, `numb_EGAT=5`
- **Training**: `epochs=100`, `lr=0.006`, `batch_size=80`, `optimizer=AdamW`, `scheduler=ReduceLROnPlateau`
- **Target**: `target_index=0` selects which material property to predict

CLI flags override the defaults for `epochs`, `batch_size`, and `lr`.

## File Structure Notes

### Core Package (`crysco/`)
- `crysco/models/CrysCo.py`: Main model architecture
- `crysco/models/EGAT.py`, `transformer.py`, `SE.py`, `MLP.py`: Model components
- `crysco/data/data.py`: Dataset classes, loaders, `drop_last=True` on train loader to avoid BatchNorm errors with size-1 batches
- `crysco/utils/utils.py`: General utilities (graph construction, smearing, dictionary loading)
- `crysco/utils/utils_train.py`: Training loops, model setup, evaluation

### Preprocessing (`scripts/preprocessing/`)
- `data_preparation.py`: Main CIF → .pt pipeline (CLI script)
- `graph_dihedral.py`: `Graph` class computing bond angles (`angle_cosines`) and dihedral angles (`dihedral_angles`)
- `extracted_features.py`: Handcrafted per-structure features via matminer/pymatgen

### Data & Config
- `material/`: Sample CIF files and CSV data (committed for reproducibility)
- `dictionary_default.json`: 114-dim per-element feature dictionary (atomic number → vector)
- `mat2vec.csv`: Element embeddings used by the transformer encoder

### Entry Points
- `main.py`: Training script with full CLI
- `prediction.ipynb`: Inference notebook

## Important Implementation Details

### Angle Features (210-dim)
The model expects `data.angle_fea` of shape `[n_atoms, 210]`. This is constructed in `data_preparation.py` by:
1. Running the `Graph` class from `graph_dihedral.py` on each pymatgen structure
2. Flattening `graph.angle_cosines` `[n_atoms, 12, 12]` → `[n_atoms, 144]`
3. Concatenating with `graph.dihedral_angles` `[n_atoms, 66]`
4. Result: `[n_atoms, 210]` which matches the `MLP([210, dim1, dim1])` input in `CrysCo.py`

### Output Directory Convention
PyTorch Geometric's `InMemoryDataset` expects the `.pt` file inside `<data_dir>/processed/`. When calling `data_preparation.py`, pass the output path as `<dir>/processed/<name>.pt` — the script will save there and training will find it via `get_dataset(data_dir, filename)`.

### Device Handling
`main.py` accepts `--device cuda|cpu` and auto-falls-back to CPU if CUDA is unavailable. All downstream calls (model setup, training, evaluation) use the same device.

### Small Dataset Handling
The train loader uses `drop_last=True` to avoid BatchNorm errors when the last batch has only 1 sample. For tiny datasets, set `--batch_size` to equal the number of training samples (e.g., `--batch_size 3` with 3 training samples).

## Dependencies

Key packages required (see `requirements.txt`):
- PyTorch 2.0+ & PyTorch Geometric
- ASE (Atomic Simulation Environment)
- pymatgen, matminer
- NumPy, Pandas, SciPy
- scikit-learn

## Misc

- Random seed is set via `--seed` (default 42) for reproducibility
- Model checkpoints are saved as `.pth`
- Graph construction includes 4-body interactions via dihedral angles
