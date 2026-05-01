import torch, numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax
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
import torch
from torch import nn
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch, numpy as np
from torch import Tensor
import torch.nn as nn
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=16,
        heads=4
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads
        self.to_qkv = nn.Linear(dim, dim_inner * 3)

    def forward(self, x, mask=None):
        has_degree_m_dim = x.ndim == 4

        if has_degree_m_dim:
            x = rearrange(x, '... 1 -> ...')

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        if mask is not None:
            mask = rearrange(mask, 'b n -> b 1 n 1')
            k = k.masked_fill(~mask, -torch.finfo(q.dtype).max)
            v = v.masked_fill(~mask, 0.)

        k = k.softmax(dim=-2)
        q = q.softmax(dim=-1)

        kv = torch.einsum('bhnd, bhne -> bhde', k, v)
        out = torch.einsum('bhde, bhnd -> bhne', kv, q)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if has_degree_m_dim:
            out = rearrange(out, '... -> ... 1')

        return out

class GatedGCN(MessagePassing):

    def __init__(self, node_dim=64, edge_dim=64, epsilon=1e-5):
        super().__init__(aggr='add')
        self.W_src  = nn.Linear(node_dim, node_dim)
        self.W_dst  = nn.Linear(node_dim, node_dim)
        self.W_A    = nn.Linear(node_dim, edge_dim)
        self.W_B    = nn.Linear(node_dim, edge_dim)
        self.W_C    = nn.Linear(edge_dim, edge_dim)
        self.act    = nn.SiLU()
        self.sigma  = nn.Sigmoid()
        self.norm_x = nn.LayerNorm([node_dim])
        self.norm_e = nn.LayerNorm([edge_dim])
        self.eps    = epsilon

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_src.weight); self.W_src.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_dst.weight); self.W_dst.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_A.weight);   self.W_A.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_B.weight);   self.W_B.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_C.weight);   self.W_C.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index

        # Calculate gated edges
        sigma_e = self.sigma(edge_attr)
        e_sum   = scatter(src=sigma_e, index=i , dim=0)
        e_gated = sigma_e / (e_sum[i] + self.eps)

        # Update the nodes (this utilizes the gated edges)
        out = self.propagate(edge_index, x=x, e_gated=e_gated)
        out = self.W_src(x) + out
        out = x + self.act(self.norm_x(out))

        # Update the edges
        edge_attr = edge_attr + self.act(self.norm_e(self.W_A(x[i]) \
                                                   + self.W_B(x[j]) \
                                                   + self.W_C(edge_attr)))

        return out, edge_attr

    def message(self, x_j, e_gated):
        return e_gated * self.W_dst(x_j)
        

#this code is obtained from  https://www.cell.com/patterns/pdf/S2666-3899(22)00076-9.pdf      
class EGAT_att(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, dropout_rate,fc_layers=2):
        super(EGAT_att, self).__init__()

        self.act       = act
        self.fc_layers = fc_layers
        self.global_mlp   = torch.nn.ModuleList()
        self.bn_list      = torch.nn.ModuleList()   

        for i in range(self.fc_layers + 1):
            if i == 0:
                lin    = torch.nn.Linear(dim+108, dim)
                self.global_mlp.append(lin)       
            else: 
                if i != self.fc_layers :
                    lin = torch.nn.Linear(dim, dim)
                else:
                    lin = torch.nn.Linear(dim, 1)
                self.global_mlp.append(lin)     
            bn = BatchNorm1d(dim)
            self.bn_list.append(bn)     

    def forward(self, x, batch, glbl_x):
        out   = torch.cat([x,glbl_x],dim=-1)
        for i in range(0, len(self.global_mlp)):
            if i   != len(self.global_mlp) -1:
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)    
            else:
                out = self.global_mlp[i](out)   
                out = tg_softmax(out,batch)                
        return out

        x           = getattr(F, self.act)(self.node_layer1(chunk))
        x           = self.atten_layer(x)
        out         = tg_softmax(x,batch)
        return out


class EGAT_LAYER(MessagePassing):
    def __init__(self, dim, activation, use_batch_norm, dropout, fc_layers=2, **kwargs):
        super().__init__(aggr='add', flow='target_to_source', **kwargs)
        self.activation_func = getattr(F, activation)
        self.dropout = dropout
        self.dim = dim
        self.heads = 4
        self.weight = Parameter(torch.Tensor(dim * 2, self.heads * dim))
        self.attention = Parameter(torch.Tensor(1, self.heads, 2 * dim))
        self.bias = Parameter(torch.Tensor(dim)) if kwargs.get('add_bias', True) else None
        self.bn = nn.BatchNorm1d(self.heads)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        combined_x_i = self.activation_func(torch.matmul(torch.cat([x_i, edge_attr], dim=-1), self.weight)).view(-1, self.heads, self.dim)
        combined_x_j = self.activation_func(torch.matmul(torch.cat([x_j, edge_attr], dim=-1), self.weight)).view(-1, self.heads, self.dim)
        alpha = self.activation_func((torch.cat([combined_x_i, combined_x_j], dim=-1) * self.attention).sum(dim=-1))
        if self.bn:
            alpha = self.activation_func(self.bn(alpha))
        alpha = tg_softmax(alpha, edge_index_i)
        return (combined_x_j * F.dropout(alpha, p=self.dropout, training=self.training).view(-1, self.heads, 1)).transpose(0, 1)

    def update(self, aggr_out):
        return aggr_out.mean(dim=0) + (self.bias if self.bias is not None else 0)
class EGATs_attention(MessagePassing):
    def __init__(self, dim, activation='relu', use_batch_norm=False, dropout_rate=0.5, 
                 num_heads=4, add_bias=True, num_fc_layers=2, edge_dim=None, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.dim = dim
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.add_bias = add_bias
        self.num_fc_layers = num_fc_layers
        self.edge_dim = edge_dim if edge_dim is not None else dim

        self.bn1 = nn.BatchNorm1d(num_heads) if use_batch_norm else None
        self.W = Parameter(torch.Tensor(dim * 2, num_heads * dim))
        self.att = Parameter(torch.Tensor(1, num_heads, 2 * dim))
        if add_bias:
            self.bias = Parameter(torch.Tensor(dim))
        else:
            self.bias = None

        self.edge_transform = nn.Linear(1, self.edge_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.edge_transform.weight)
        if self.edge_transform.bias is not None:
            nn.init.zeros_(self.edge_transform.bias)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device)
        out, edge_attr_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out, edge_attr_updated

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        out_i = torch.cat([x_i, edge_attr], dim=-1)
        out_j = torch.cat([x_j, edge_attr], dim=-1)

        act_func = getattr(F, self.activation)
        out_i = act_func(torch.matmul(out_i, self.W))
        out_j = act_func(torch.matmul(out_j, self.W))

        out_i = out_i.view(-1, self.num_heads, self.dim)
        out_j = out_j.view(-1, self.num_heads, self.dim)

        alpha = (torch.cat([out_i, out_j], dim=-1) * self.att).sum(dim=-1)
        if self.use_batch_norm and self.bn1 is not None:
            alpha = self.bn1(alpha)
        alpha = act_func(alpha)
        alpha = tg_softmax(alpha, edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        alpha_avg = alpha.mean(dim=1, keepdim=True)
        self.edge_attr_updated = self.edge_transform(alpha_avg)

        out_j = (out_j * alpha.view(-1, self.num_heads, 1)).transpose(0, 1)
        return out_j

    def update(self, aggr_out, edge_attr):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out, self.edge_attr_updated
