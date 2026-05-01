import torch, numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric

from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
import torch
from torch import nn
import numpy as np
import pandas as pd

import torch
from torch import nn
class ResidualSE3(nn.Module):
    def forward(self, x, res):
        out = {}
        for degree, tensor in x.items():
            degree = str(degree)
            out[degree] = tensor
            if degree in res:
                out[degree] = out[degree] + res[degree]
        return out

class LinearSE3(nn.Module):
    def __init__(
        self,
        fiber_in,
        fiber_out
    ):
        super().__init__()
        self.weights = nn.ParameterDict()

        for (degree, dim_in, dim_out) in (fiber_in & fiber_out):
            key = str(degree)
            self.weights[key]  = nn.Parameter(torch.randn(dim_in, dim_out) / sqrt(dim_in))

    def forward(self, x):
        out = {}
        for degree, weight in self.weights.items():
            out[degree] = einsum('b n d m, d e -> b n e m', x[degree], weight)
        return out

class NormSE3(nn.Module):
    def __init__(
        self,
        input_dim,
        nonlin=nn.GELU(),
        gated_scale=False,
        eps=1e-12,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.nonlin = nonlin
        self.eps = eps

        # Define fiber based on input dimension
        self.fiber = [('degree1', input_dim)]  # Example: [('degree1', 64)]

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for degree, chan in self.fiber:
            self.transform[str(degree)] = nn.ParameterDict({
                'scale': nn.Parameter(torch.ones(1, 1, chan)) if not gated_scale else None,
                'w_gate': nn.Parameter(torch.rand(chan, chan).uniform_(-1e-3, 1e-3)) if gated_scale else None
            })

    def forward(self, x):
        # Compute the norms and normalized features
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        phase = x / norm

        outputs = []
        for degree, chan in self.fiber:
            # Transform on norms
            parameters = self.transform[str(degree)]
            gate_weights, scale = parameters['w_gate'], parameters['scale']

            transformed = norm.view(*norm.shape, 1)

            if scale is None:
                scale = torch.einsum('bnd, de -> bne', transformed, gate_weights)

            transformed = self.nonlin(transformed * scale)
            transformed = transformed.squeeze(-1)  # Remove the last dimension of size 1

            # Nonlinearity on norm
            output = (transformed * phase)
            outputs.append(output)

        # Concatenate outputs along the last dimension
        concatenated_outputs = torch.cat(outputs, dim=-1)  # Shape: (num_points, num_degrees * dimensionality)
        desired_tensor = torch.mean(concatenated_outputs, dim=0)

        return desired_tensor
