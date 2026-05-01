"""
CrysCo: Hybrid Graph-Transformer for Materials Property Prediction

This package contains the core components of the CrysCo model including
graph neural networks, transformers, and data processing utilities.
"""

__version__ = "0.1.0"
__author__ = "Mohammad Madani"
__email__ = "mohammad73madani73@gmail.com"

from .models.CrysCo import CrysCo
from .models.MLP import MLP
from .data.data import setup_data_loaders, get_dataset, StructureDataset
from .utils.utils_train import train_model, evaluate

__all__ = [
    'CrysCo',
    'MLP',
    'setup_data_loaders',
    'get_dataset',
    'StructureDataset',
    'train_model',
    'evaluate'
]