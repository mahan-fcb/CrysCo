"""
Multi-Layer Perceptron (MLP) Module for CrysCo

This module provides a flexible MLP implementation used throughout
the CrysCo model architecture for various feature transformations.
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable activation functions.

    Args:
        hidden_sizes: List of integers specifying layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function to use between layers (default: None)
    """

    def __init__(self, hidden_sizes, act=None):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = act

        num_layers = len(hidden_sizes)
        layers = []

        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            # Add activation between layers (but not after the final layer)
            if (act is not None) and (i < num_layers - 2):
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the MLP."""
        return self.mlp(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(hidden_sizes={self.hidden_sizes}, activation={self.activation})'