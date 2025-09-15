import random
import itertools
import numpy as np

from rdkit.Geometry import Point3D
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom

from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

from MetalloGen import chem, process
from MetalloGen import globalvars as gv
from MetalloGen.utils import ic

def get_transition_metal_center(geometry_name):
    
    tm_center = {
        '6_octahedral': 26,
        '4_tetrahedral': 30,
        '4_square_planar': 78,
    }
    return tm_center.get(geometry_name,26)

def get_dummy_center_for_valid(geometry_name):
    # Get the dummy center to get valid ace molecule
    dummy_centers = {
        '2': chem.Atom('O'),
        '3': chem.Atom('P'),
        '4': chem.Atom('Si'),
        '5': chem.Atom('P'),
        '6': chem.Atom('S'),
    }
    steric_number = geometry_name.split('_')[0]
    return dummy_centers[steric_number] if steric_number in dummy_centers else chem.Atom('Fe')

def get_dummy_center(geometry_name):
    # Get the dummy center to embed the molecule
    dummy_centers = {
        '2_linear': chem.Atom('Fe'),
        '2_bent_135': chem.Atom('S'),
        '2_bent_90': chem.Atom('S'),
        '3_t_shaped': chem.Atom('P'),
        '3_trigonal_planar': chem.Atom('Al'),
        '3_trigonal_pyramidal': chem.Atom('P'),
        '4_seesaw': chem.Atom('P'),
        '4_square_planar': chem.Atom('Pt'),
        '4_tetrahedral': chem.Atom('Si'),
        '5_pentagonal_planar': chem.Atom('Pd'),
        '5_square_pyramidal': chem.Atom('Fe'),
        '5_trigonal_bipyramidal': chem.Atom('P'),
        '6_hexagonal_planar': chem.Atom('Fe'),
        '6_octahedral': chem.Atom('Fe'),
        '6_pentagonal_pyramidal': chem.Atom('Fe'),
    }
    return dummy_centers[geometry_name] if geometry_name in dummy_centers.keys() else chem.Atom('Fe')


def get_dummy_atom_list(bond_num):
    # Get the dummy atom for the conformer
    dummy_atom = {
        1: [chem.Atom('O'),chem.Atom('Ag')], 
        2: [chem.Atom('Al'),chem.Atom('P')], 
        3: [chem.Atom('Si')],
        4: [chem.Atom('P')],
        5: [chem.Atom('Fe'),chem.Atom('S')],
        6: [chem.Atom('Fe')],
    }
    if bond_num in dummy_atom:
        return dummy_atom[bond_num]
    else:
        return [chem.Atom('Fe')]


def check_dummy_atom(binding_indices, binding_infos, option):
    if option == 0:
        return len(binding_indices) > 1 # Only for haptic

    elif option == 1:
        return len(binding_indices) > 1 or len(binding_infos) == 1 # Haptic + single atom bonded 

    else:
        return True


def initialize_molecule_properties(metal_complex, option):
        
    total_atom_num = 1
    total_wo_dummy = 1
    total_chg = 0
    total_mult = metal_complex.multiplicity
    
    metal_index = metal_complex.metal_index
    ligands = metal_complex.ligands
    atom_indices_for_each_ligand = metal_complex.get_atom_indices_for_each_ligand()
    
    for ligand in ligands:
        total_atom_num += len(ligand.molecule.atom_list)
        total_wo_dummy += len(ligand.molecule.atom_list)
        
    total_atom_list = [None] * total_atom_num
    
    for i in range(len(ligands)):
        ligand = ligands[i]
        atom_indices = atom_indices_for_each_ligand[i]
        atom_list = ligand.molecule.atom_list
        binding_infos = ligand.binding_infos
        
        total_chg += ligand.molecule.chg
        
        for j in range(len(atom_list)):
            total_atom_list[atom_indices[j]] = atom_list[j].copy()
            
        for j in range(len(binding_infos)):
            binding_indices = binding_infos[j][0]
            if check_dummy_atom(binding_indices, binding_infos, option):
                total_atom_num += 1        
                
    return total_atom_num, total_wo_dummy, total_chg, total_mult, total_atom_list  
    
def get_alternative_molecule(metal_complex, option):
    """
    Returns the alternative molecule for the conformer, with dummy atoms and dummy center,
    
    params: metal_complex: complex_geometry.MetalComplex object
            distance: float
            option: True: set dummy atoms selectively, False: set dummy atoms uniformly
    
    returns: ace_mol: chem.Molecule object
    """
    metal_index = metal_complex.metal_index
    ligands = metal_complex.ligands
    atom_indices_for_each_ligand = metal_complex.get_atom_indices_for_each_ligand()
    adj_matrix = metal_complex.get_adj_matrix()
    binding_groups = metal_complex.get_binding_groups()
    
    geometry = metal_complex.geometry_type
    geometry_name = geometry.geometry_name

    # Initialize molecule properties
    dummy_center = get_dummy_center(geometry_name)
    dummy_indices = [] # Indices of dummy atom
    metal_binding_infos = [] # The final atom connecting to the metal atom (Because non-hatpic ligands)
    
    total_atom_num, total_wo_dummy, total_chg, total_mult, total_atom_list = initialize_molecule_properties(metal_complex, option)
    total_adj = np.zeros((total_atom_num, total_atom_num))
    
    m = 0
    
    dummy_atom_cn_list = []
     
    for i in range(len(ligands)):
        ligand = ligands[i]
        atom_indices = atom_indices_for_each_ligand[i]
                
        binding_infos = ligand.binding_infos
        atom_list = ligand.molecule.atom_list
        mol_adj = ligand.molecule.adj_matrix
         
        for j in range(len(atom_list)):
            for k in range(len(atom_list)):
                total_adj[atom_indices[j]][atom_indices[k]] = mol_adj[j][k]
                total_adj[atom_indices[k]][atom_indices[j]] = mol_adj[k][j]

        for j in range(len(binding_infos)):
            binding_info = binding_infos[j]
            binding_indices, binding_site = binding_info
            is_multidentate = len(binding_infos) > 1
            is_haptic = len(binding_indices) > 1
            if check_dummy_atom(binding_indices, binding_infos, option):
                for k in range(len(binding_indices)):
                    idx = atom_indices[binding_indices[k]]
                    total_adj[idx][total_wo_dummy+m] = 1
                    total_adj[total_wo_dummy+m][idx] = 1
                dummy_atom_cn_list.append(len(binding_indices))
                dummy_indices.append(total_wo_dummy+m)
                metal_binding_infos.append([total_wo_dummy+m,binding_site,is_multidentate,is_haptic])
                m += 1
            else:
                idx = atom_indices[binding_indices[0]]
                metal_binding_infos.append([idx,binding_site,is_multidentate,is_haptic])

    total_atom_list[metal_index] = get_dummy_center_for_valid(geometry_name)
    
    # Connect center_binding atoms to the center
    for metal_binding_info in metal_binding_infos:
        binding_index = metal_binding_info[0]
        total_adj[binding_index][metal_index] = 1
        total_adj[metal_index][binding_index] = 1
        
    n = len(total_adj)
    
    # Make alternative molecules ...

    ace_mol_list = []
    total_dummy_atom_list = []
    if len(dummy_atom_cn_list) > 0:
        cn_dict = dict()
        dummy_atom_candidate_list = []
        for cn in dummy_atom_cn_list:
            if cn not in cn_dict:
                cn_dict[cn] = len(cn_dict)
                dummy_atom_candidate_list.append(get_dummy_atom_list(cn))
        dummy_atom_candidate_list = list(itertools.product(*dummy_atom_candidate_list))
        for candidate_list in dummy_atom_candidate_list:
            total_dummy_atom_list.append([candidate_list[cn_dict[cn]] for cn in dummy_atom_cn_list])
    else:
        total_dummy_atom_list.append([])

    for dummy_atom_list in total_dummy_atom_list:
        ace_mol = chem.Molecule()
        ace_mol.chg = total_chg
        ace_mol.multiplicity = total_mult
        ace_mol.atom_list = total_atom_list + [atom.copy() for atom in dummy_atom_list]
        ace_mol.adj_matrix = total_adj
        
        groups = process.group_molecules(total_adj)

        
        transition_metal_z_list = gv.transition_metal_z_list
        metal_indices = [metal_index]

        try:
            valid_ace_mol = ace_mol.get_valid_molecule(False, method='pulp',MetalCenters=metal_indices)
        except:
            valid_ace_mol = None

        if valid_ace_mol is None:
            try:
                valid_ace_mol = ace_mol.get_valid_molecule(False, method='xyz2mol',MetalCenters=metal_indices)
            except:
                continue

        valid_ace_mol.chg = total_chg
        valid_ace_mol.multiplicity = total_mult

        valid_ace_mol.atom_list[metal_index] = dummy_center
        
        ace_mol_list.append(valid_ace_mol)
    
    return ace_mol_list, dummy_indices, metal_binding_infos

def get_repulsive_potential(coordinate_list, d_criteria=0.5, p=6):
    distance_matrix = cdist(coordinate_list, coordinate_list)
    potential_matrix = 1/(distance_matrix-d_criteria)**p
    return np.sum(potential_matrix)

def align_double_single_ligand(metal_complex, positions, d_criteria=1.7):
    
    ligands = metal_complex.ligands
    atom_indices_for_each_ligand = metal_complex.get_atom_indices_for_each_ligand()
    
    #ratio_criteria = 0.6
    #d_criteria = 0.5

    for i in range(len(ligands)):
                
        tmp_positions = positions.copy()
        
        ligand = ligands[i]
        atom_indices = atom_indices_for_each_ligand[i]
        binding_infos = ligand.binding_infos
        atom_list = ligand.molecule.atom_list
        
        #print(binding_infos)
        if len(binding_infos) > 1 or len(binding_infos[0][0]) > 1:
            continue
        
        binding_indices, binding_site = binding_infos[0]
        adj_matrix = ligand.get_adj_matrix()
        binding_neighbors = np.where(adj_matrix[binding_indices[0],:] == 1)[0]
        
        l_positions = tmp_positions[atom_indices,:].copy()

        # First align
        v1 = l_positions[binding_indices[0],:]
        v2 = metal_complex.geometry_type.direction_vector[binding_site-1]
        
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        angle = np.arccos(np.dot(v1,v2)) 
        axis = np.cross(v1,v2)
        
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
            r = R.from_rotvec(angle * axis)
            l_positions = r.apply(l_positions)
        
        # Shift to zero
        d = l_positions[binding_indices[0],:]
        l_positions = l_positions - l_positions[binding_indices[0],:]
        
        # Second align
        v1 = np.mean(l_positions[binding_neighbors,:],axis=0)
        v2 = metal_complex.geometry_type.direction_vector[binding_site-1]
        
        if np.linalg.norm(v1) < 1e-6:
            continue
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        angle = np.arccos(np.dot(v1,v2)) - 0.03 # Nearly 180o
        axis = np.cross(v1,v2)
        
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
            r = R.from_rotvec(angle * axis)
            l_positions = r.apply(l_positions)
            
        # Shift back
        l_positions = l_positions + d
        tmp_positions[atom_indices,:] = l_positions
        
        # Check the validity of the new positions
        # Check the distance between the ligands
        distance_matrix = cdist(tmp_positions,tmp_positions)
        other_indices = np.setdiff1d(np.arange(len(tmp_positions)),atom_indices)
        for j in atom_indices:
            for k in other_indices:
                if distance_matrix[j,k] < d_criteria:
                    #print('Invalid alignment ... atoms are too close ...')
                    continue
       
        # Check potential
        current_potential = get_repulsive_potential(positions)
        new_potential = get_repulsive_potential(tmp_positions)
        if new_potential > 10 * current_potential:
            #print('Invalid alignment ... potential is too high ...')
            continue
        
        positions = tmp_positions.copy()
                
    
    return positions

def get_embedding(metal_complex, scale=1.0, option=0, align=False, use_random=True):
    
    atom_d_criteria = 0.5
    ratio_criteria  = 0.65
    adj_ratio_criteria = 1.4
    
    # Make a conformer based on the representation
    new_complex = metal_complex.copy()
    total_atom_list = new_complex.get_atom_list()
    
    # new_complex gives the geometry ... (metal complex remains no change)

    alternative_ace_mol_list, dummy_indices, metal_binding_infos = get_alternative_molecule(new_complex,option)

    metal_index = new_complex.metal_index
    metal_r = new_complex.center_atom.get_radius()
    
    radius_list = [atom.get_radius() for atom in total_atom_list] # Before alteration of atoms ...
    n = len(radius_list)
    R = np.repeat(np.array(radius_list),n).reshape((n,n))
    R = R + R.T
    
    # Make cmap/params ...
    direction_vector = new_complex.geometry_type.direction_vector
    params = rdDistGeom.EmbedParameters()
    params.useRandomCoords = True
    params.maxIterations = 100
    params.useBasicKnowledge = False
    params.ignoreSmoothingFailures = True

    cmap = dict()
    success = True
    total_adj = alternative_ace_mol_list[0].adj_matrix
    
    if use_random is True:
        params.randomSeed = random.randint(0,1000000)
    
    #print("Alternative molecules embedding ...")
    
    candidate_list = []
    scales_for_haptic = [0.4,0.5,0.6,0.7]

    haptic_exist = False

    for alternative_ace_mol in alternative_ace_mol_list:
        
        idx = alternative_ace_mol_list.index(alternative_ace_mol) 
        rd_mol = alternative_ace_mol.get_rd_mol()
        print("Trying ", Chem.MolToSmiles(rd_mol))

        positions = None

        # Try many scale values for haptic (haptic embedding seems to work differently) 
        for haptic_scale in scales_for_haptic:
            
            failed = False
            # Make cmap for embedding 
            for metal_binding_info in metal_binding_infos:
                binding_index, binding_site, is_multidentate, is_haptic = metal_binding_info # list, int, bool, bool
                atom_r = alternative_ace_mol_list[0].atom_list[binding_index].get_radius()
                if is_haptic:
                    distance = (metal_r + atom_r) * haptic_scale
                    haptic_exist = True
                else:
                    distance = (metal_r + atom_r) * scale
                x, y, z = direction_vector[binding_site-1] * distance
                cmap[binding_index] = Point3D(x,y,z)
            cmap[metal_index] = Point3D(0.0,0.0,0.0)  
            params.SetCoordMap(cmap)

            try:
                AllChem.EmbedMolecule(rd_mol,params)
            except:
                try:
                    geometry_name = new_complex.geometry_type.geometry_name
                    temp_metal_num = get_transition_metal_center(geometry_name)
                    alternative_ace_mol.atom_list[metal_index].set_atomic_number(temp_metal_num) 
                    AllChem.EmbedMolecule(rd_mol,params)
                except:
                    print("Embedding failed ...")
                    failed = True
            try:
                conformer = rd_mol.GetConformer()
            except:
                if not failed:
                    print("Conformer not obtained ...")
                    failed = True
            try:
                positions = conformer.GetPositions()
            except:
                if not failed:
                    print("Position not obtained ...")
                    failed = True

            if failed or positions is None:
                # Try different molecule ...
                alternative_ace_mol = alternative_ace_mol.get_valid_molecule(False, method='xyz2mol',MetalCenters=[metal_index])
                rd_mol = alternative_ace_mol.get_rd_mol()
                print("Trying ", Chem.MolToSmiles(rd_mol))
                
                try:
                    AllChem.EmbedMolecule(rd_mol,params)
                except:
                    print("Embedding failed ...")
                    continue
                try:  
                    conformer = rd_mol.GetConformer()
                except:
                    print("Conformer not obtained ...")
                    continue
                try:
                    positions = conformer.GetPositions()
                except:
                    print("Position not obtained ...")
                    continue

            if not failed:
                break

            # No need to perform multiple embedding ...
            if not haptic_exist:
                break

        # If position does not exist, move on to the next alternative ace mol
        if positions is None:
            continue 
    
        # Update bond distances that are too long ... (Because of atom replacement)
        q_updates = dict()
        bond_scale = 1.0
        bond_list = np.stack(np.where(new_complex.get_adj_matrix()>0),axis=1) # Not for dummy atom ...
        for bond in bond_list:
            s, e = bond
            if s > e:
                continue
            if s == 0 or e == 0: # Not for metal ...
                continue 
            d = ic.get_distance(positions[:metal_complex.num_atom], s, e)
            # Check ratio ...          
            r_sum = radius_list[s] + radius_list[e]
            ratio = d/r_sum
            delta_ratio = ratio - bond_scale
            if delta_ratio > 0.2: # If too long ...
                q_updates[(s,e)] = - delta_ratio * r_sum
            else:
                q_updates[(s,e)] = 0.0
        ic.update_xyz(positions[:metal_complex.num_atom],q_updates) # Others are the same ...

        if align is True:
            positions = align_double_single_ligand(new_complex, positions)
        
        # Check the validity of initial embedding ...
        # Check the distance matrix between different atoms ... (first criterion)
        distance_matrix = cdist(positions[:metal_complex.num_atom],positions[:metal_complex.num_atom])
        np.fill_diagonal(distance_matrix,1e6)
        if np.any(distance_matrix < atom_d_criteria):
            print("Atoms are too close ...")
            candidate_list.append((positions,-100000))
            continue
        
        # Check the collapse between ligands with ratio method (second criterion)
        ratio_matrix = distance_matrix/R
        min_ratio = np.min(ratio_matrix)
        if min_ratio < ratio_criteria:
            print("Atoms are too close ...")
            candidate_list.append((positions,-50000))
            continue

        # Finally, check the ratios that are ambiguous, between the ligands ...
        adj_matrix = np.where(distance_matrix/R < adj_ratio_criteria, 1, 0)
        original_adj_matrix = metal_complex.get_adj_matrix()
        # Remove bonds between the metal ...
        adj_matrix[metal_index,:] = 0.0
        adj_matrix[:,metal_index] = 0.0
        original_adj_matrix[metal_index,:] = 0.0
        original_adj_matrix[:,metal_index] = 0.0

        diff = np.sum(np.abs(adj_matrix - original_adj_matrix))

        if diff > 0:
            print ("Undesired bond is detected ...")
            candidate_list.append((positions,-diff))
            continue

        print("Embedding success!")
        
        return positions[:metal_complex.num_atom]
    
    print("Embedding failed ...")

    if len(candidate_list) == 0:
        return None
    else:
        # Get the best position among the position list ... (Check by value ...)
        maximum_value = -100000000
        final_positions = candidate_list[0][0]
        for candidate in candidate_list:
            value = candidate[1]
            if value > maximum_value:
                maximum_value = value
                final_positions = candidate[0]
        print(f"Returning the best position ... maximum value {maximum_value}")
        return final_positions[:metal_complex.num_atom]

