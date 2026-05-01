"""
CrysCo: Hybrid Graph-Transformer Model for Materials Property Prediction

This module implements the main CrysCo model architecture that combines
Edge Graph Attention Networks (EGAT), Transformer layers, and SE(3)-equivariant
components for predicting inorganic material properties.
"""

import torch
import numpy as np
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GCNConv,
    DiffGroupNorm
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
import pandas as pd

from .MLP import MLP
from .transformer import Transformer, ElementEncoder, PositionalEncoder
from .SE import ResidualSE3, NormSE3, LinearSE3
from .EGAT import LinearAttention, GatedGCN, EGAT_att, EGAT_LAYER, EGATs_attention
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
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

class CrysCo(torch.nn.Module):
    def __init__(self, data,
        out_dims=64,
        d_model=512,
        N=3,
        heads=4,
        compute_device=None,
        dim1=64,
        dim2=150,
        numb_embbeding=1,
        numb_EGAT=5,
        numb_GATGCN=5,
        batch_norm =True,
        pool="global_add_pool",
        act="silu",     

        dropout_rate =0,
        **kwargs
    ):
        super(CrysCo, self).__init__()
        self.dropout_rate =dropout_rate
        self.human_embedding = nn.Linear(24, dim1)
        self.human_bn = nn.BatchNorm1d(dim1)
        self.batch_norm =batch_norm
        self.pool         = pool
        self.act          = act
        self.out_dims = out_dims
        self.d_model = d_model
        self.out_hidden = [1024, 512, 256, 128]
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.edge_att = EGAT_att(dim1, act, batch_norm,  dropout_rate)
        self.encoder = Transformer(d_model=self.d_model,N=self.N,heads=self.heads)
        self.resnet = ResidualNN(self.d_model, self.out_dims, self.out_hidden)
        output_dim = 1 if data[0].y.ndim == 0 else len(data[0].y)
        self.pre_lin_list_E = torch.nn.ModuleList()
        self.pre_lin_list_N_angle = torch.nn.ModuleList()
        self.pre_lin_list_N = torch.nn.ModuleList()
        for i in range(numb_embbeding):
            embed_atm = nn.Sequential(MLP([114,dim1, dim1],  act=nn.SiLU()), nn.LayerNorm(dim1))
            self.pre_lin_list_N.append(embed_atm)
            embed_bnd = nn.Sequential(MLP([50,dim1,dim1], act=nn.SiLU()), nn.LayerNorm(dim1))
            self.pre_lin_list_E.append(embed_bnd)    
            embed_ang = nn.Sequential(MLP([210,dim1,dim1], act=nn.SiLU()), nn.LayerNorm(dim1))
            self.pre_lin_list_N_angle.append(embed_ang)
    
        self.conv_list = torch.nn.ModuleList()
        for i in range(numb_GATGCN):
            conv = EGATs_attention(dim1)
            self.conv_list.append(conv)
        self.conv1_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(numb_EGAT):
            conv1 = EGAT_LAYER(dim1, act, batch_norm,  dropout_rate)
            self.conv1_list.append(conv1)
            bn = BatchNorm1d(dim1)
            self.bn_list.append(bn)
        self.lin_out = torch.nn.Linear(dim1*3, output_dim)   

    def forward(self, data):
   
        for i in range(1):
            out_xa = self.pre_lin_list_N_angle[i](data.angle_fea)
            out_xa = F.softplus(out_xa)
            out_x = self.pre_lin_list_N[i](data.x)
            out_x = F.softplus(out_x)
            out_e = self.pre_lin_list_E[i](data.edge_attr)
            out_e = F.softplus(out_e)
        prev_out_x = out_x
        out_xs,edge_attr_x = self.conv_list[0](out_xa,data.edge_index, out_e)
        for i in range(0, len(self.conv_list)):
            out_xs,edge_attr_x = self.conv_list[i](out_xs,data.edge_index, edge_attr_x)
        out_e=out_e+edge_attr_x
        for i in range(0, len(self.conv_list)):
            out_x = self.conv1_list[i](out_x, data.edge_index, out_e)
            out_x = self.bn_list[i](out_x)         
            out_x = torch.add(out_x, prev_out_x)
        out_x = F.dropout(out_x, p=0.1, training=self.training)


        out_a = self.edge_att(out_x,data.batch,data.glob_feat)#obtained from deeperGATGNN model
        out_x = (out_x)*out_a    
        output = self.encoder(data.src.to(dtype=torch.long,non_blocking=True), data.frac.to(dtype=torch.float,non_blocking=True))
        mask = (data.src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.resnet(output)
        output = output.masked_fill(mask, 0)
        output = output.sum(dim=1)/(~mask).sum(dim=1)    
        out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)
        human_fea = self.human_embedding(torch.tensor(data.human_d, dtype=torch.float32).to(out_x.device))
        human_fea = self.human_bn(human_fea)
        out_x = torch.cat((out_x,human_fea,output),1)
        out       = self.lin_out(out_x)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
