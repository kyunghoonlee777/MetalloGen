import numpy as np
import pickle

import geometries as geom
from scipy.spatial.transform import Rotation as R
from itertools import permutations

from MetalloGen import chem, process
from MetalloGen.utils import shape

def get_normalized_vector(v):
    d = np.linalg.norm(v)
    if d < 1e-6:
        print ('Almost zero vector detected !!!')
        return v
    else:
        return v/d

def get_ligand_center(molecule,ligand_site):
    site_coordinates = []
    coordinate_list = molecule.get_coordinate_list()
    for idx in ligand_site:
        coordinate = coordinate_list[idx]
        site_coordinates.append(coordinate)
    site_coordinates = np.array(site_coordinates)
    return np.mean(site_coordinates,axis=0)

def get_geometry_info(molecule, metal_index,ligand_sites):
    vectors = []
    metal_centers = []
    metal_coordinate = molecule.atom_list[metal_index].get_coordinate()
    for ligand_site in ligand_sites:
        center_coordinate = get_ligand_center(molecule,ligand_site)
        vector = center_coordinate - metal_coordinate
        vector = get_normalized_vector(vector)
        vectors.append(vector)
    return vectors

def get_ligand_sites(molecule,metal_z_list):
    
    adj_matrix = np.copy(molecule.get_adj_matrix())
    metal_indices = []
    for i,atom in enumerate(molecule.atom_list):
        atomic_number = atom.get_atomic_number()
        if atomic_number in metal_z_list:
            metal_indices.append(i)
    # Disconnect center and ligand atoms
    connecting_sites = set()
    for i in metal_indices:
        bonding_sites = np.where(adj_matrix[i] > 0)[0].tolist()
        adj_matrix[i,bonding_sites] = 0
        adj_matrix[bonding_sites,i] = 0
        connecting_sites = connecting_sites | set(bonding_sites)
    # Make ligand molecules
    ligand_groups = process.group_molecules(adj_matrix)
    # Remove metal center ...
    index = 0
    while index < len(ligand_groups):
        removed = False
        if len(ligand_groups[index]) == 1:
            for metal_index in metal_indices:
                if metal_index == ligand_groups[index][0]:
                    del(ligand_groups[index])
                    removed = True
                    break
        if not removed:
            index += 1

    #all_indices = list(connecting_sites | set(metal_indices))
    # Find ligand sites
    all_indices = list(connecting_sites) # All ligand connecting sites
    all_indices.sort()
    index_function = dict()
    for i, index in enumerate(all_indices):
        index_function[i] = index
    reduce_function = np.ix_(all_indices,all_indices)
    reduced_adj_matrix = adj_matrix[reduce_function]
    groups = process.group_molecules(reduced_adj_matrix)
    ligand_sites = []
    for group in groups:
        atom_indices = [index_function[j] for j in group]
        ligand_sites.append(atom_indices)
    ligands = []
    ligand_binding_infos = []
    for ligand_group in ligand_groups:
        ligand = chem.Molecule()
        ligand.atom_list = [molecule.atom_list[j] for j in ligand_group]
        reduce_function = np.ix_(ligand_group,ligand_group)
        ligand.adj_matrix = adj_matrix[reduce_function]
        reduce_function = dict()
        binding_info = []
        for i in range(len(ligand_group)):
            reduce_function[ligand_group[i]] = i
        for binding_site in ligand_sites:
            if binding_site[0] in ligand_group:
                binding_info.append([reduce_function[j] for j in binding_site])
        ligands.append(ligand)
        ligand_binding_infos.append(binding_info)
    return ligands,ligand_binding_infos,ligand_sites, metal_indices

def calculate_shape_similarity(geometry_info,vector_infos_dict):
    
    rmsd_dict = dict()
    
    for key in vector_infos_dict:
        vectors = vector_infos_dict[key]
        vectors = np.array([v/np.linalg.norm(v) for v in vectors])
        rmsd, _ = shape.shape_measure(geometry_info,vectors)
        
        if rmsd > 1000:
            continue
        
        rmsd_dict[key] = rmsd
        
    return rmsd_dict

def process_oms(om_info_list, metal_z_list):
    '''
    om_infos: om_name -> pkl: z_list, adj_matrix, chg_list, coords, metal_idx
                     -> result: metal_element, steric_number, geometry (name or array), ligand ids, min_rmsd 
    ligand_infos: ligand_id -> pkl: z_list, adj_matrix, chg (chg_list cannot be obtained), coords, binding_indices
                           -> result: is_haptic, denticity, type (Ex. 3_2) 
    '''
    om_infos = dict()
    for om_info in om_info_list:
        om_name, om = om_info
        ligands, ligand_binding_infos, ligand_sites, metal_indices = get_ligand_sites(om, metal_z_list)

        if len(metal_indices) != 1:
            continue
        # Check steric number
        steric_number = len(ligand_sites)
        if steric_number < 2:
            #print ('Low steric number',om_name,ligand_sites)
            continue

        metal_index = metal_indices[0]
        metal = om.atom_list[metal_index].get_element()
        min_rmsd = 100000

        geometry_info = get_geometry_info(om,metal_index,ligand_sites)

        if steric_number in geometries.keys():
            vector_infos_dict = {key: geometry_vectors[key] for key in geometries[steric_number]}
            rmsd_dict = calculate_shape_similarity(geometry_info,vector_infos_dict)
            if len(rmsd_dict) > 0:
                geometry_info = min(rmsd_dict,key=rmsd_dict.get)
                min_rmsd = rmsd_dict[geometry_info]  

        z_list = om.get_z_list()
        adj_matrix = om.get_adj_matrix()
        chg_list = om.get_chg_list()
        coords = om.get_coordinate_list()
        om_result = dict()
        om_result['pkl'] = [z_list, adj_matrix, chg_list, coords, metal_index]
        om_result['result'] = [metal, steric_number, geometry_info, [], min_rmsd]
        om_infos[om_name] = om_result
        
        # Collect ligand information ...
        for i, ligand in enumerate(ligands):
            connectivity_id = ligand.get_connectivity_id()
            binding_info = ligand_binding_infos[i]
            ligand_id = f'{connectivity_id}_{len(binding_info)}' # denticity
            om_result['result'][-2].append(ligand_id)
            if ligand_id not in ligand_infos:
                # Make ligand result
                z_list = ligand.get_z_list()
                adj_matrix = ligand.get_adj_matrix()
                chg = None
                coords = ligand.get_coordinate_list()
                ligand_result = dict()
                hapticity_infos = []
                is_haptic = False
                for binding in binding_info:
                    hapticity_infos.append(len(binding))
                    if len(binding) > 1:
                        is_haptic = True                
                denticity = len(hapticity_infos)
                ligand_result['pkl'] = [z_list, adj_matrix, None, coords, binding_info]
                ligand_result['result'] = [is_haptic, denticity, hapticity_infos]
                ligand_infos[ligand_id] = ligand_result 
        

    return om_infos, ligand_infos

def main(args):
    import sys
    import os

    data_directory = args.data_directory
    num_core = args.num_core
    file_size = args.file_size
    
    raw_directory = os.path.join(data_directory,'om/om_raw/')
    om_save_directory = os.path.join(data_directory,'om/om_tmp/')
    ligand_save_directory = os.path.join(data_directory,'ligand/ligand_tmp/')
    
    if not os.path.exists(raw_directory):
        print (f'Raw data directory {raw_directory} does not exist! Please run extract.py first ...')
        sys.exit()
    
    metal_z_list = list(range(3,4)) + list(range(11,14)) + list(range(19,32)) + list(range(37,51)) + list(range(55,84)) + list(range(87,113))
        
    file_names = os.listdir(input_directory)   

    geometries = geom.SN_known_geometry_dict.copy()
    geometry_vectors = geom.known_geometry_vector_dict.copy()
    
    om_infos = dict()
    ligand_infos = dict()
 
    os.makedirs(om_save_directory,exist_ok=True)
    os.makedirs(ligand_save_directory,exist_ok=True)

    # Collect total om info ...
    om_list = []
    for file_name in file_names:
        with open(os.path.join(raw_directory,file_name),'rb') as f:
            for om_info in pickle.load(f):
                om_name, z_list, adj_matrix, chg_list, coords = om_info
                om = chem.Molecule()
                om.atom_list = [chem.Atom(z) for z in z_list]
                om.adj_matrix = adj_matrix
                om.atom_feature['chg'] = chg_list
                process.locate_molecule(om, coords)
                om_list.append((om_name, om))
             
    # Multiprocess om infos ...
    if num_core > 1:
        import multiprocessing

        params = []
        n = len(om_list)
        interval = int(len(om_list)/num_core) + 1
        for i in range(0,n,interval):
            start = i
            end = min(i+interval,n)
            params.append((om_list[start:end],metal_z_list))
        with multiprocessing.Pool(num_core) as p:
            results = p.starmap(process_oms, params)
        om_infos = dict()
        ligand_infos = dict()
        for result in results:
            om_infos.update(result[0])
            ligand_infos.update(result[1])
             
    else:
        om_infos, ligand_infos = process_oms(om_list, metal_z_list)

    print (len(om_infos),'Organometallic complexes were extracted ...')
    print ('Saving data ...')

    # Save om info in tmp file ...
    keys = list(om_infos.keys())
    n = len(keys)
    for i in range(0,n,file_size):
        start = i
        end = min(i+file_size,n)
        sub_om_infos = dict()
        for j in range(start,end):
            sub_om_infos[keys[j]] = om_infos[keys[j]]
        with open(os.path.join(om_save_directory,f'om_info_{start}_{end}.pkl'),'wb') as f:
            pickle.dump(sub_om_infos,f)

    # Save ligand info in tmp file ...        
    keys = list(ligand_infos.keys())
    n = len(keys)
    for i in range(0,n,file_size):
        start = i
        end = min(i+file_size,n)
        sub_ligand_infos = dict()
        for j in range(start,end):
            sub_ligand_infos[keys[j]] = ligand_infos[keys[j]]
        with open(os.path.join(ligand_save_directory,f'ligand_info_{start}_{end}.pkl'),'wb') as f:
            pickle.dump(sub_ligand_infos,f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract organometallic complexes from raw data.")
    parser.add_argument("--data_directory", "-dd", type=str, help="Data directory containing proceessed data files.", required=True)
    parser.add_argument("--num_core", "-n", type=int, help="Number of cores to use", default=1)
    parser.add_argument("--file_size", "-s", type=int, help="Interval to save intermediate results", default=10000)
    
    args = parser.parse_args()
    main(args)
    