import collections
import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch_geometric.data import DataLoader, Dataset
from composition import _element_composition
import os
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
from scipy import interpolate
##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch.linalg import cross
import os
import torch
from torch_geometric.data import DataLoader

class CompositionError(Exception):
    """Exception class for composition errors"""
    pass

def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    regex = r"([A-Z][a-z]*)\s*([-*\.\d]*)"
    for m in re.finditer(regex, f):
        el = m.group(1)
        amt = float(m.group(2)) if m.group(2) else 1
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError(f'{f} is an invalid formula!')
    return sym_dict

def parse_formula(formula):
    formula = formula.replace('@', '').replace('[', '(').replace(']', ')')
    regex = r"\(([^\(\)]+)\)\s*([\.\d]*)"
    m = re.search(regex, formula)
    if m:
        factor = float(m.group(2)) if m.group(2) else 1
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join([f"{el}{amt}" for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    return get_sym_dict(formula, 1)

def _element_composition(formula):
    elmap = parse_formula(formula)
    return {k: v for k, v in elmap.items() if abs(v) >= 1e-6}


class EDMDataset(Dataset):
    def __init__(self, data):
        self.X = torch.as_tensor(data[0], dtype=torch.float32)
        self.y = torch.as_tensor(data[1], dtype=torch.float32)
        self.formula = data[2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.formula[idx]

def get_edm(path, n_elements=16, scale=True):
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    if isinstance(path, str):
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    else:
        df = path

    df['count'] = [len(_element_composition(form)) for form in df['formula']]
    #df = df.groupby(by='formula').mean().reset_index()  # mean of duplicates

    list_ohm = [OrderedDict(_element_composition(form)) for form in df['formula']]
    list_ohm = [OrderedDict(sorted(mat.items(), key=lambda x:-x[1])) for mat in list_ohm]

    y = df['target'].values.astype(np.float32)
    formula = df['formula'].values

    edm_array = np.zeros(shape=(len(list_ohm), n_elements, len(all_symbols) + 1), dtype=np.float32)
    elem_num = np.zeros(shape=(len(list_ohm), n_elements), dtype=np.float32)
    elem_frac = np.zeros(shape=(len(list_ohm), n_elements), dtype=np.float32)

    for i, comp in enumerate(tqdm(list_ohm, desc="Generating EDM", unit="formulae")):
        for j, (elem, count) in enumerate(list_ohm[i].items()):
            if j == n_elements:
                break
            edm_array[i, j, all_symbols.index(elem) + 1] = count
            elem_num[i, j] = all_symbols.index(elem) + 1

    if scale:
        for i in range(edm_array.shape[0]):
            frac = edm_array[i, :, :].sum(axis=-1) / edm_array[i, :, :].sum(axis=-1).sum()
            elem_frac[i, :] = frac
    else:
        for i in range(edm_array.shape[0]):
            frac = edm_array[i, :, :].sum(axis=-1)
            elem_frac[i, :] = frac

    if n_elements == 16:
        n_elements = np.max(np.sum(elem_frac > 0, axis=1, keepdims=True))
        elem_num = elem_num[:, :n_elements]
        elem_frac = elem_frac[:, :n_elements]

    elem_num = elem_num.reshape(elem_num.shape[0], elem_num.shape[1], 1)
    elem_frac = elem_frac.reshape(elem_frac.shape[0], elem_frac.shape[1], 1)
    out = np.concatenate((elem_num, elem_frac), axis=1)

    return out, y, formula

class EDM_CsvLoader():
    def __init__(self, csv_data, batch_size=64, pin_memory=True, n_elements=6, scale=True):
        self.main_data = get_edm(csv_data, n_elements=n_elements, scale=scale)
        self.batch_size = batch_size
        self.pin_memory = pin_memory
    def get_data_loaders(self):
        dataset = EDMDataset(self.main_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=self.pin_memory)
        return loader


def create_global_feat(atoms_index_arr):
    comp = np.bincount(atoms_index_arr, minlength=108) / len(atoms_index_arr)
    return comp.reshape(1, -1)

def process_data(data_path, processed_path, processing_args):
    print("Processing data to:", os.path.join(data_path, processed_path))
    assert os.path.exists(data_path), f"Data path not found in {data_path}"

    if processing_args["dictionary_source"] != "generated":
        atom_dictionary = get_dictionary(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dictionary_default.json"))

    target_property_file = os.path.join(data_path, processing_args["target_path"])
    assert os.path.exists(target_property_file), f"Targets not found in {target_property_file}"

    with open(target_property_file) as f:
        target_data = list(csv.reader(f))

    data_list = []
    for index, target_row in enumerate(target_data):
        structure_id = target_row[0]
        data = Data()

        if processing_args["data_format"] != "db":
            ase_crystal = ase.io.read(os.path.join(data_path, f"{structure_id}.{processing_args['data_format']}"))
        else:
            ase_crystal = ase_crystal_list[index]

        length = len(ase_crystal)
        elements = list(set(ase_crystal.get_chemical_symbols()))

        distance_matrix = ase_crystal.get_all_distances(mic=True)
        distance_matrix_trimmed = threshold_sort(distance_matrix, processing_args["graph_max_radius"], processing_args["graph_max_neighbors"], adj=False)

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        if self_loops := True:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=length, fill_value=0)
            distance_matrix_mask = (distance_matrix_trimmed.fill_diagonal_(1) != 0).int()
        elif self_loops := False:
            distance_matrix_mask = (distance_matrix_trimmed != 0).int()

        data.edge_index = edge_index
        data.edge_weight = edge_weight
        data.edge_descriptor = {"distance": edge_weight, "mask": distance_matrix_mask}

        target = target_row[1:]
        y = torch.Tensor(np.array([target], dtype=np.float32))
        data.y = y

        atoms_index = ase_crystal.get_atomic_numbers()
        gatgnn_glob_feat = create_global_feat(atoms_index)
        data.glob_feat = torch.Tensor(np.repeat(gatgnn_glob_feat, len(atoms_index), axis=0)).float()

        z = torch.LongTensor(atoms_index)
        data.z = z

        u = torch.zeros((3,))
        data.u = u.unsqueeze(0)

        data.structure_id = [[structure_id] * length]

        if processing_args["verbose"] == "True" and ((index + 1) % 500 == 0 or (index + 1) == len(target_data)):
            print("Data processed:", index + 1, "out of", len(target_data))

        data_list.append(data)

    n_atoms_max = max(len(ase_crystal) for ase_crystal in ase_crystal_list)
    species = sorted(set(sum([list(set(ase_crystal.get_chemical_symbols())) for ase_crystal in ase_crystal_list], [])))

    if processing_args["verbose"] == "True":
        print("Max structure size:", n_atoms_max, "Max number of elements:", len(species))
        print("Unique species:", species)

    if processing_args["dictionary_source"] != "generated":
        for data in data_list:
            atom_fea = np.vstack([atom_dictionary[str(atom)] for atom in data.z])
            data.x = torch.Tensor(atom_fea)
    elif processing_args["dictionary_source"] == "generated":
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        lb.fit(species)
        for data in data_list:
            data.x = torch.Tensor(lb.transform(data.ase.get_chemical_symbols()))

    for data in data_list:
        data = OneHotDegree(data, processing_args["graph_max_neighbors"] + 1)

    if processing_args["edge_features"] == "True":
        distance_gaussian = GaussianSmearing(0, 1, processing_args["graph_edge_length"], 0.2)
        NormalizeEdge(data_list, "distance")
        for data in data_list:
            data.edge_attr = distance_gaussian(data.edge_descriptor["distance"])
            if processing_args["verbose"] == "True" and ((index + 1) % 500 == 0 or (index + 1) == len(target_data)):
                print("Edge processed:", index + 1, "out of", len(target_data))

    Cleanup(data_list, ["ase", "edge_descriptor"])

    processed_full_path = os.path.join(data_path, processed_path)
    os.makedirs(processed_full_path, exist_ok=True)

    if processing_args["dataset_type"] == "inmemory":
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), os.path.join(processed_full_path, "data.pt"))
    elif processing_args["dataset_type"] == "large":
        for i, data in enumerate(data_list):
            torch.save(data, os.path.join(processed_full_path, f"data_{i}.pt"))

def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.where(
        mask,
        np.nan,
        rankdata(matrix * (-1 if reverse else 1), method="ordinal", axis=1)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i, row in enumerate(distance_matrix_trimmed):
            temp = np.where(row != 0)[0]
            adj_list[i, :len(temp)] = temp
            adj_attr[i, :len(temp)] = matrix[i, temp]
        return np.where(distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix), adj_list, adj_attr
    else:
        return np.where(distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix)
    
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super().__init__()
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", torch.linspace(start, stop, resolution))

    def forward(self, dist):
        return torch.exp(self.coeff * torch.pow(dist.unsqueeze(-1) - self.offset.view(1, -1), 2))


def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = F.one_hot(degree(idx, data.num_nodes, dtype=torch.long), num_classes=max_degree + 1).to(torch.float)
    data.x = torch.cat([x.view(-1, 1) if x.dim() == 1 else x, deg.to(x.dtype)], dim=-1) if x is not None and cat else deg
    return data


def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        return json.load(f)


def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except AttributeError:
                pass


def bond_angles(bond_vec, edge_index_bnd_ang):
    bond_vec /= torch.linalg.norm(bond_vec, dim=-1, keepdim=True)
    i, j = edge_index_bnd_ang
    cos_ang = (bond_vec[i] * bond_vec[j]).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    sin_ang = cos_ang.acos().sin()
    return torch.hstack([cos_ang, sin_ang])


def dihedral_angles(pos, edge_index_bnd, edge_index_dih_ang):
    dih_idx = edge_index_bnd.T[edge_index_dih_ang.T].reshape(-1, 4).T
    i, j, k, l = dih_idx
    u1, u2, u3 = pos[j] - pos[i], pos[k] - pos[j], pos[l] - pos[k]
    u1, u2, u3 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True), u2 / torch.linalg.norm(u2, dim=-1, keepdim=True), u3 / torch.linalg.norm(u3, dim=-1, keepdim=True)
    cos_ang = (cross(u1, u2) * cross(u2, u3)).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    sin_ang = (u1 * cross(u2, u3)).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    return torch.hstack([cos_ang, sin_ang])


def add_edges(edge_index, new_edge_index, num_nodes):
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    A[edge_index.split(1)] = True
    A[new_edge_index.split(1)] = True
    return A.nonzero().T


def add_edges_v2(edge_index, new_edge_index):
    mask = (edge_index[:, None, :] - new_edge_index[..., None]).abs().sum(0).gt(0).all(1)
    return torch.hstack([edge_index, new_edge_index[:, mask]])


def edge_set(edge_index):
    return set(map(tuple, edge_index.T.tolist()))


def mask_edges(edge_index, edge_index_to_mask, num_nodes):
    M = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    M[edge_index_to_mask.split(1)] = True
    return M[edge_index.split(1)].flatten()

def periodic_radius_graph(pos, box, cutoff):
    num_atoms, num_dims = pos.shape
    shape = (box // cutoff).to(torch.long)
    cell_dims = box / shape
    num_cells = shape.prod()

    coords = (pos / cell_dims).to(torch.long)
    shifts = cell_shift(coords, shape)

    pos -= box * shifts
    coords = wrap_cell_coord(coords, shape)

    cumprod = torch.cat((torch.tensor([1], device=shape.device), shape[:-1].cumprod(dim=0)))

    indices = (cumprod * coords).sum(dim=-1)

    max_atoms_per_cell = torch.bincount(indices).max()
    cells = -torch.ones(num_cells, max_atoms_per_cell, dtype=torch.long, device=pos.device)
    counts = torch.zeros(num_cells, dtype=torch.long)
    for atom_idx, cell_idx in enumerate(indices):
        n = counts[cell_idx]
        cells[cell_idx, n] = atom_idx
        counts[cell_idx] += 1

    center_coords = torch.unique(coords, dim=0)
    center_indices = (cumprod * center_coords).sum(dim=-1)
    nbr_disp = torch.cartesian_prod(*torch.arange(-1, 2).expand(num_dims, 3)).to(pos.device)
    nbr_coords = center_coords.unsqueeze(1) + nbr_disp.unsqueeze(0)
    nbr_shifts = cell_shift(nbr_coords, shape)
    nbr_coords = wrap_cell_coord(nbr_coords, shape)
    nbr_indices = (cumprod * nbr_coords).sum(dim=-1)

    ii, jj = torch.cartesian_prod(torch.arange(max_atoms_per_cell), torch.arange(max_atoms_per_cell)).T.to(pos.device)

    src, dst, vec = [], [], []
    for c1, c2, s in zip(center_indices, nbr_indices, nbr_shifts):
        i = cells[c1].unsqueeze(0).expand(len(nbr_disp), -1)
        j = cells[c2]
        i = torch.take_along_dim(i, ii[None, :], dim=-1)
        j = torch.take_along_dim(j, jj[None, :], dim=-1)
        v = (pos[j] + s.unsqueeze(1) * box) - pos[i]

        i, j, v = i.reshape(-1), j.reshape(-1), v.reshape(-1, num_dims)

        mask = torch.logical_and(i != -1, j != -1)
        src.append(i[mask])
        dst.append(j[mask])
        vec.append(v[mask])

    src = torch.cat(src)
    dst = torch.cat(dst)
    vec = torch.cat(vec, dim=0)

    mask = vec.norm(dim=-1) < cutoff
    src, dst, vec = src[mask], dst[mask], vec[mask]

    mask = src != dst
    src, dst, vec = src[mask], dst[mask], vec[mask]

    return torch.stack((src, dst)), vec


def GetRanges(dataset, descriptor_label):
    edge_desc = torch.cat([data.edge_descriptor[descriptor_label] for data in dataset if len(data.edge_descriptor[descriptor_label]) > 0])
    feature_min, feature_max = edge_desc.min(), edge_desc.max()
    mean, std = edge_desc.mean(), edge_desc.std()
    return mean, std, feature_min, feature_max


def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)
    for data in dataset:
        data.edge_descriptor[descriptor_label] = (data.edge_descriptor[descriptor_label] - feature_min) / (feature_max - feature_min)


class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data



