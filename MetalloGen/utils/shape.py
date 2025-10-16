import numpy as np
from scipy.spatial.transform import Rotation as R
from itertools import permutations

from MetalloGen import process
from MetalloGen import globalvars as gv

def get_binding_sites(adj_matrix, metal_index):
    adj_matrix = np.array(adj_matrix)
    connected_indices = np.where(adj_matrix[metal_index] == 1)[0]
    index_function = dict()
    for i, index in enumerate(connected_indices):
        index_function[i] = index
    reduct_function = np.ix_(connected_indices, connected_indices)
    reduced_adj_matrix = adj_matrix[reduct_function]
    groups = process.group_molecules(reduced_adj_matrix)
    binding_sites = []
    for group in groups:
        binding_sites.append([index_function[i] for i in group])
    
    return binding_sites

def shape_measure(binding_vectors, direction_vectors, k=5):
    
    N = len(binding_vectors)
    if N != len(direction_vectors):
        raise ValueError("Number of binding vectors and direction vectors do not match")
    
    binding_vectors = np.array(binding_vectors)
    selected_binding_vectors = binding_vectors[:min(k, N)]
    direction_vectors = np.array(direction_vectors)
    direction_indices = list(range(N))
    selected_direction_indices = list(permutations(direction_indices, min(k, N)))
    
    min_rmsd = 1e6
    best_assigned_indices = None
    
    for perm in selected_direction_indices:
        selected_direction_vectors = direction_vectors[list(perm)]
        r, rssd = R.align_vectors(selected_direction_vectors, selected_binding_vectors)
        
        rotated_binding_vectors = r.apply(binding_vectors)
        
        assigned_indices = []
        for v in rotated_binding_vectors:
            similarities = np.dot(direction_vectors, v)
            assigned_index = np.argmax(similarities)
            assigned_indices.append(assigned_index)

        if len(set(assigned_indices)) != N:
            continue
        
        assigned_direction_vectors = direction_vectors[assigned_indices]
        diff = np.linalg.norm(assigned_direction_vectors - rotated_binding_vectors)
        rmsd = np.sqrt(np.mean(diff**2))
        
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            best_assigned_indices = assigned_indices
        
    return min_rmsd, best_assigned_indices

def assign_shape(ace_mol, center_index):
    adj_matrix = ace_mol.get_adj_matrix()
    coordinate_list = ace_mol.get_coordinate_list()
    
    binding_sites = get_binding_sites(adj_matrix, center_index)
    binding_vectors = []
    for binding_site in binding_sites:
        binding_vector = np.mean(coordinate_list[binding_site], axis=0) - coordinate_list[center_index]
        binding_vector = binding_vector / np.linalg.norm(binding_vector)
        binding_vectors.append(binding_vector)
    
    CN = len(binding_sites)
    candidate_shapes = gv.CN_known_geometries_dict[CN]
    candidate_direcotions = [gv.known_geometries_vector_dict[shape] for shape in candidate_shapes]
    
    min_rmsd = 1e6
    best_geometry = None
    best_assigned_indices = None
    for i in range(len(candidate_shapes)):
        shape = candidate_shapes[i]
        direction_vectors = candidate_direcotions[i]
        rmsd, assigned_indices = shape_measure(binding_vectors, direction_vectors)
        print(shape, rmsd)
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            best_geometry = shape
            best_assigned_indices = assigned_indices
    
    if best_geometry is None:
        raise ValueError("Cannot find the best geometry")
    
    assigned_result = []
    for i in range(len(assigned_indices)):
        assigned_result.append((binding_sites[i], best_assigned_indices[i]))
    
    return best_geometry, assigned_result

