import os
import csv
import numpy as np
from matminer.datasets import load_dataset
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN

def human_features(directory_path, csv_file):
    os.chdir(directory_path)
    # Read the target property file
    with open(csv_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader]
    target_data = target_data[1:]

    # Create featurizers
    density_featurizer = DensityFeatures()
    global_symmetry_featurizer = GlobalSymmetryFeatures()
    voronoi_nn = VoronoiNN()

    # Process structure files and calculate features
    data_list = []
    error_indices = []

    for index in range(len(target_data)):
        structure_id = target_data[index][0]
        cif_file = os.path.join(directory_path, structure_id + ".cif")

        if os.path.exists(cif_file):
            structure = Structure.from_file(cif_file)

            try:
                # Compute density features
                density_features = density_featurizer.featurize(structure)

                # Compute global symmetry features
                global_symmetry_features = global_symmetry_featurizer.featurize(structure)

                volume = structure.volume
                primitive_structure = SpacegroupAnalyzer(structure).find_primitive()
                num_atoms_primitive_cell = len(primitive_structure)
                num_atoms = len(structure)
                volume_per_atom = volume / num_atoms

                atom_radii = [element.atomic_radius for element in structure.species]
                total_atom_volume = sum((4/3) * np.pi * (radius**3) for radius in atom_radii)
                packing_fraction = total_atom_volume / volume
                lattice_parameters = structure.lattice.parameters

                voronoi_coord_numbers = [len(voronois) for voronois in voronoi_nn.get_all_voronoi_polyhedra(structure)]

                mean_voronoi_coord_number = np.mean(voronoi_coord_numbers)
                std_dev_voronoi_coord_number = np.std(voronoi_coord_numbers)

                bond_angles = []
                bond_lengths = []
                neighbor_distances = []

                for i, site in enumerate(structure):
                    neighbors = voronoi_nn.get_nn_info(structure, i)
                    for neighbor_info in neighbors:
                        central_coord = site.coords
                        neighbor_coord = neighbor_info['site'].coords

                        # Compute bond lengths and distances
                        bond_length = np.linalg.norm(central_coord - neighbor_coord)
                        bond_lengths.append(bond_length)
                        neighbor_distance = np.linalg.norm(central_coord - neighbor_coord)
                        neighbor_distances.append(neighbor_distance)

                        # Compute bond angles
                        for second_neighbor_info in neighbors:
                            if second_neighbor_info != neighbor_info:
                                second_neighbor_coord = second_neighbor_info['site'].coords
                                bond_vector_1 = central_coord - neighbor_coord
                                bond_vector_2 = central_coord - second_neighbor_coord
                                if np.linalg.norm(bond_vector_1) != 0 and np.linalg.norm(bond_vector_2) != 0:
                                    angle = np.arccos(np.dot(bond_vector_1, bond_vector_2) /
                                                      (np.linalg.norm(bond_vector_1) * np.linalg.norm(bond_vector_2)))
                                    bond_angles.append(np.degrees(angle))

                mean_avg_bond_angle = np.nanmean(bond_angles)
                std_dev_avg_bond_angle = np.nanstd(bond_angles)
                mean_avg_bond_length = np.mean(bond_lengths)
                std_dev_avg_bond_length = np.std(bond_lengths)
                mean_neighbor_distance = np.mean(neighbor_distances)
                std_dev_neighbor_distance = np.std(neighbor_distances)
                min_neighbor_distance = np.min(neighbor_distances)
                max_neighbor_distance = np.max(neighbor_distances)

                # Extract lattice parameters
                a, b, c, alpha, beta, gamma = lattice_parameters

                # Additional global symmetry features
                gs_features = [global_symmetry_features[0], global_symmetry_features[2], global_symmetry_features[4]]

                # Concatenate all calculated values into a single vector
                structure_vector = [
                    volume, num_atoms_primitive_cell, num_atoms, volume_per_atom,
                    packing_fraction, mean_voronoi_coord_number, std_dev_voronoi_coord_number,
                    mean_avg_bond_angle, std_dev_avg_bond_angle, mean_avg_bond_length,
                    std_dev_avg_bond_length, mean_neighbor_distance, std_dev_neighbor_distance,
                    min_neighbor_distance, max_neighbor_distance,
                    a, b, c, alpha, beta, gamma,
                    *gs_features
                ]
                structure_tensor = torch.tensor(structure_vector, dtype=torch.float32)

                # Append the structure vector to the data list
                data_list.append(structure_vector)
            except RuntimeError as e:
                error_indices.append(index)
                if "QH6154 Qhull precision error" in str(e):
                    print("QH6154 Qhull precision error occurred. Skipping problematic data and continuing...",structure_id)
                    # Handle the error by skipping problematic data and continuing
                    # You can add your error handling logic here
                    pass
                else:
                    print(f"RuntimeError for structure {structure_id}: {e}")
            except TypeError as e:
                error_indices.append(index)
                print(f"TypeError for structure {structure_id}: {e}")
            except ValueError as e:
                error_indices.append(index)
                print(f"ValueError for structure {structure_id}: {e}")
            if (index + 1) % 500 == 0 or (index + 1) == len(target_data):
                print("Data processed:", index + 1, "out of", len(target_data))

    return data_list

