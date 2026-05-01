"""Smoke tests verifying the package imports correctly after reorganization."""
import os
import sys

# Add repo root to path so tests can run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model_imports():
    from crysco.models.CrysCo import CrysCo
    from crysco.models.EGAT import EGAT_att
    from crysco.models.transformer import Transformer
    from crysco.models.MLP import MLP
    assert CrysCo is not None
    assert MLP is not None


def test_data_imports():
    from crysco.data.data import setup_data_loaders, get_dataset, StructureDataset, GetY
    assert setup_data_loaders is not None
    assert get_dataset is not None


def test_training_imports():
    from crysco.utils.utils_train import (
        train_model,
        train_one_epoch,
        model_setup,
        evaluate,
        write_results,
    )
    assert train_model is not None
    assert model_setup is not None


def test_preprocessing_imports():
    from scripts.preprocessing.graph_dihedral import Graph
    g = Graph(neighbors=12, rcut=8, delta=1)
    assert g.neighbors == 12


if __name__ == "__main__":
    test_model_imports()
    test_data_imports()
    test_training_imports()
    test_preprocessing_imports()
    print("All import tests passed.")
