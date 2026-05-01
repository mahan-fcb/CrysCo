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
class ElementEncoder(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        mat2vec = 'mat2vec.csv'
        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=torch.float32)
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(dtype=torch.float32)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb

class ResidualNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super(ResidualNN, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False) if (dims[i] != dims[i+1]) else nn.Identity() for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
class PositionalEncoder(nn.Module):

    def __init__(self,d_model,resolution=100):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        x = torch.linspace(0, self.resolution - 1,self.resolution).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,self.d_model).view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x /torch.pow(50,2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.embed = ElementEncoder(d_model=self.d_model)
        self.pe = PositionalEncoder(self.d_model, resolution=5000)
        self.ple = PositionalEncoder(self.d_model, resolution=5000)

        self.emb_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.heads, dim_feedforward=2048, dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N)

    def forward(self, src, frac):
        x = self.embed(src) * 2 ** self.emb_scaler
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        pe[:, :, :self.d_model // 2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)

        return x
    
