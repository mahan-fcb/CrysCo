from time import time
from pymatgen.core.periodic_table import Element
import numpy as np
import torch
from itertools import combinations, product
import csv
import os
from time import time
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.functional as F
from torch.utils.data import Dataset
from itertools import combinations
from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element

class Graph(object):
    '''
    Graph object for creation of atomic graphs with bond and node attributes from pymatgen structure
    '''

    def __init__(
        self,
        neighbors=12,
        rcut=8,
        delta=1,
    ):

        self.neighbors = neighbors
        self.rcut = rcut
        self.delta = delta
        self.bond = []
        self.nbr = []
        self.angle_cosines = []
        self.angle_sines = []
        self.dihedral_angles = []

    def setGraphFea(self, structure):

        if self.rcut > 0:
            pass
        else:
            species = [site.specie.symbol for site in structure.sites]
            self.rcut = max(
                [Element(elm).atomic_radius * 3 for elm in species]
            )

        all_nbrs = structure.get_all_neighbors(self.rcut, include_index=True)

        len_nbrs = np.array([len(nbr) for nbr in all_nbrs])

        indexes = np.where((len_nbrs < self.neighbors))[0]

        for i in indexes:
            cut = self.rcut
            curr_N = len(all_nbrs[i])
            while curr_N < self.neighbors:
                cut += self.delta
                nbr = structure.get_neighbors(structure[i], cut)
                curr_N = len(nbr)
            all_nbrs[i] = nbr

        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        self.nbr = torch.LongTensor(
            [
                list(map(lambda x: x[2], nbrs[: self.neighbors]))
                for nbrs in all_nbrs
            ]
        )
        self.bond = torch.Tensor(
            [
                list(map(lambda x: x[1], nbrs[: self.neighbors]))
                for nbrs in all_nbrs
            ]
        )

        cart_coords = torch.Tensor(np.array(
            [structure[i].coords for i in range(len(structure))]
        ))
        atom_nbr_fea = torch.Tensor(np.array(
            [
                list(map(lambda x: x[0].coords, nbrs[: self.neighbors]))
                for nbrs in all_nbrs
            ]
        ))
        centre_coords = cart_coords.unsqueeze(1).expand(
            len(structure), self.neighbors, 3
        )
        dxyz = atom_nbr_fea - centre_coords
        r = self.bond.unsqueeze(2)
        self.angle_cosines = torch.matmul(
            dxyz, torch.swapaxes(dxyz, 1, 2)
        ) / torch.matmul(r, torch.swapaxes(r, 1, 2))

        # Calculate sine of bond angles
       # dxyz_reshaped = dxyz.unsqueeze(2).repeat(1, 1, self.neighbors, 1)

# Compute cross product
       # cross_product = torch.cross(dxyz_reshaped, torch.swapaxes(dxyz, 1, 2))
       # sine_denominator = torch.norm(cross_product, dim=2)
       # self.angle_sines = torch.clamp(sine_denominator / (r ** 2), min=-1.0, max=1.0)
        self.calculate_dihedral_angles(structure)

    def calculate_dihedral_angles(self, structure):
        """
        Calculates dihedral angles for each set of four atoms and structures them into a consistent shape.
        """
        max_dihedrals_per_atom = self.neighbors * (self.neighbors - 1) // 2
        dihedral_angles = torch.zeros((len(structure), max_dihedrals_per_atom), dtype=torch.float)

        for i in range(len(structure)):
            angles = []
            nbr_indices = self.nbr[i].tolist()
            for j, k, l in combinations(nbr_indices, 3):
                if j != i and k != i and l != i and j != k and j != l and k != l:
                    p0, p1, p2, p3 = [structure[idx].coords for idx in [i, j, k, l]]
                    angle = self.calculate_single_dihedral_angle(torch.tensor(p0), torch.tensor(p1), torch.tensor(p2), torch.tensor(p3))
                    if not np.isnan(angle):  # Check if angle calculation is valid
                        angles.append(angle)

            # Fill the tensor for the current atom with calculated angles up to max_dihedrals_per_atom
            dihedral_angles[i, :len(angles)] = torch.tensor(angles, dtype=torch.float)[:max_dihedrals_per_atom]

        self.dihedral_angles = dihedral_angles

    def calculate_single_dihedral_angle(self, p0, p1, p2, p3):
        """
        Safely calculate a single dihedral angle given four points.
        """
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2

        n1 = torch.cross(b0, b1)
        n2 = torch.cross(b1, b2)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        n1_u = n1 / (torch.norm(n1) + epsilon)
        n2_u = n2 / (torch.norm(n2) + epsilon)

        # Calculate the angle using atan2 for a full range angle
        x = torch.dot(n1_u, n2_u)
        m1 = torch.cross(n1_u, b1 / (torch.norm(b1) + epsilon))
        y = torch.dot(m1, n2_u)

        angle = torch.atan2(y, x)

        return angle.item()

class load_graphs_targets(object):

    '''
    structureData should
    be in dict format
                      structure:{pymatgen structure},
                      property:{}
                      formula: None or formula
    if not from database
    '''

    def __init__(self, neighbors=12, rcut=0, delta=1):

        self.neighbors = neighbors
        self.rcut = rcut
        self.delta = delta

    def load(self, data):
        structure = data["structure"]
        target = data["target"]
        # print(target)
        graph = Graph(
            neighbors=self.neighbors, rcut=self.rcut, delta=self.delta
        )
        # try:
        # print("graphs")
        graph.setGraphFea(structure)
        print(data["formula"])
        return (graph, target)
        # except:
        #    return None
        
class CrystalGraphDataset():
    '''
    A Crystal graph dataset container for genrating and loading pytorch dataset to be passed to train test and validation loader
    '''

    def __init__(
        self,
        dataset,
        neighbors=12,
        rcut=8,
        delta=1,
        mp_load=False,
        mp_pool=None,
        mp_cpu_count=None,
        **kwargs
    ):

        # ================================
        print("Loading {} graphs .......".format(len(dataset)))
        # =================================

        t1 = time()

        load_graphs = load_graphs_targets(
            neighbors=neighbors,
            rcut=rcut,
            delta=delta,
        )

        results = process(
            load_graphs.load,
            dataset,
            mp_cpu_count,
            mp_load=mp_load,
            mp_pool=mp_pool,
        )

        # print(results)

        self.graphs = [res[0] for res in results if res is not None]

       # self.targets = [
       #     torch.LongTensor(res[1]) for res in results if res is not None
        #]
        # print(self.targets)
        #self.binarizer = LabelBinarizer()
        #self.binarizer.fit(torch.cat(self.targets))

        t2 = time()
        print("Total time taken {}".format(convert(t2 - t1)))

        #self.size = len(self.targets)

    def collate(self, datalist):

        bond_feature, nbr_idx, angular_feature, dihedral, crys_idx = (
            [],
            [],
            [],
            [],
            [],
            [],
    
    
        )

        index = 0

        for (bond_fea, idx, angular_fea,ang,dih), targ in datalist:
            Natoms = bond_fea.shape[0]

            bond_feature.append(bond_fea)
            angular_feature.append(angular_fea)
            dihedral.append(dih)

            nbr_idx.append(idx + index)
            crys_idx.append([index, index + Natoms])
           # targets.append(targ)
            index += Natoms

        return (
            torch.cat(bond_feature, dim=0),
            torch.cat(angular_feature, dim=0),
            torch.cat(dihedral, dim=0),
            torch.cat(nbr_idx, dim=0),
            torch.LongTensor(crys_idx),
            
        )

    def __getitem__(self, idx):

        graph = self.graphs[idx]
        bond_feature = graph.bond
        nbr_idx = graph.nbr
        angular_feature = graph.angle_cosines
        dihedral = graph.dihedral_angles
    
       # target = self.targets[idx]

        return (bond_feature, nbr_idx, angular_feature,dihedral)


# --------------------------------------------
def process(func, tasks, n_proc, mp_load=False, mp_pool=None):

    if mp_load:
        results = []
        chunks = [tasks[i : i + n_proc] for i in range(0, len(tasks), n_proc)]
        for chunk in chunks:
            # print("chunks")
            r = mp_pool.map_async(func, chunk, callback=results.append)
            r.wait()
        mp_pool.close()
        mp_pool.join()
        return results[0]
    else:
        # print("chunks")
        return [func(task) for task in tasks]

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)

import csv
from pymatgen.io.cif import CifParser
import os

# Define your dataset loader function
def load_dataset(directory, target_property_file, neighbors=12, rcut=3, delta=1):
    loader = load_graphs_targets(neighbors=neighbors, rcut=rcut, delta=delta)
    dataset = []
    target_file_path = os.path.join(directory, target_property_file)

    with open(target_file_path) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader][1:]  # Skip header

    for row in target_data:
        structure_id = row[0]
        cif_file_path = os.path.join(directory, structure_id + ".cif")
        # Load structure data from the CIF file
        parser = CifParser(cif_file_path)
        structure = parser.get_structures()[0]  # Assuming there's only one structure in the CIF file
        # Here, assuming the target property is in the second column of the CSV file
        target = float(row[1])  # Convert the target value from string to float
        # Load graph features and target
        a = {"structure": structure, "target": target, "formula": structure_id + ".cif"}
        
        dataset.append(a)
    
    return dataset

