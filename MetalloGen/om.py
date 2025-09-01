import embed
import globalvars as gv
import chem, process

import numpy as np
import copy


class Geometry:
    
    def __init__(self, geometry_name):
        
        self.geometry_name = geometry_name # str
        self.direction_vector = gv.known_geometries_vector_dict[geometry_name] if geometry_name in gv.known_geometries_vector_dict.keys() else []
        self.permutations = gv.known_geometries_permutation_dict[geometry_name] if geometry_name in gv.known_geometries_permutation_dict.keys() else []
 
    def get_steric_number(self):
        return len(self.direction_vector)


class MetalComplex:

    def __init__(self, geometry_name, center_atom, ligands, chg, multiplicity):

        self.geometry_type = Geometry(geometry_name) # Geometry Object
        self.center_atom = center_atom # chem.Atom
        self.ligands = ligands # [Ligand Object]
        
        self.atom_indices_for_each_ligand = [] # [[int]]
        self.metal_index = None # int
        self.adj_matrix = None # np.array
        self.bo_matrix = None # np.array
        self.is_actinide = False

        num_atom = 0
        if center_atom is not None:
            num_atom += 1
        for ligand in ligands:
            num_atom += len(ligand.molecule.atom_list)

        self.num_atom = num_atom
        self.chg = chg
        self.multiplicity = multiplicity
        self.name = None


    def get_atom_indices_for_each_ligand(self):
        if len(self.atom_indices_for_each_ligand) > 0:
            return self.atom_indices_for_each_ligand
        else:
            ligands = self.ligands
            atom_indices_for_each_ligand = []
            n = 1
            for ligand in ligands:
                m = len(ligand.molecule.atom_list)
                atom_indices = [i for i in range(n,n+m)]
                atom_indices_for_each_ligand.append(atom_indices)
                n += m
            self.atom_indices_for_each_ligand = atom_indices_for_each_ligand
            return atom_indices_for_each_ligand


    def get_binding_groups(self):
        ligands = self.ligands
        atom_indices_for_each_ligand = self.get_atom_indices_for_each_ligand()
        steric_number = self.geometry_type.get_steric_number()
        n = len(ligands)
        binding_groups = [None] * steric_number
        for i in range(n):
            ligand = ligands[i]
            atom_indices = atom_indices_for_each_ligand[i]
            binding_infos = ligand.binding_infos
            for binding_info in binding_infos:
                binding_indices, binding_site = binding_info
                if binding_site is None:
                    print ("Coordination Information is not determined !!!")
                    return []
                binding_groups[binding_site-1] = [atom_indices[index] for index in binding_indices]

        return binding_groups
            

    def get_adj_matrix(self):
        metal_index = self.metal_index
        ligands = self.ligands
        atom_indices_for_each_ligand = self.get_atom_indices_for_each_ligand()
        m = len(ligands)

        n = 1
        for ligand in ligands:
            n += len(ligand.molecule.atom_list)
        
        adj_matrix = np.zeros((n,n))      
        
        for i in range(m):
            ligand = ligands[i]
            atom_indices = atom_indices_for_each_ligand[i]
            mol_adj = ligand.molecule.get_adj_matrix()
            for j in range(len(mol_adj)):
                for k in range(len(mol_adj)):
                    start = atom_indices[j]
                    end = atom_indices[k]
                    adj_matrix[start][end] = mol_adj[j][k]
                    adj_matrix[end][start] = mol_adj[k][j]
        binding_groups = self.get_binding_groups()   
        #print ("binding_groups:",binding_groups)
        for group in binding_groups:
            for i in group:
                adj_matrix[metal_index][i] = 1
                adj_matrix[i][metal_index] = 1
        return adj_matrix


    def get_atom_list(self):
        metal_index = self.metal_index
        ligands = self.ligands
        atom_indices_for_each_ligand = self.get_atom_indices_for_each_ligand()
        
        n = 1
        for ligand in ligands:
            n += len(ligand.molecule.atom_list)
        
        atom_list = [None] * n
        atom_list[metal_index] = self.center_atom.copy()
        
        for i in range(len(ligands)):
            ligand = ligands[i]
            atom_indices = atom_indices_for_each_ligand[i]
            ligand_atom_list = ligand.molecule.atom_list
            for j in range(len(ligand_atom_list)):
                atom_list[atom_indices[j]] = ligand_atom_list[j].copy()
                
        return atom_list


    def get_molecule(self):
        molecule = chem.Molecule()
        chg = self.chg
        mult = self.multiplicity
        
        atom_list = self.get_atom_list()
        adj_matrix = self.get_adj_matrix()
                
        molecule.atom_list = atom_list
        molecule.atom_feature = dict()
        molecule.adj_matrix = adj_matrix
        molecule.chg = chg
        molecule.multiplicity = mult
        
        return molecule


    def get_position(self):
        atom_list = self.get_atom_list()
        positions = [[atom.x, atom.y, atom.z] for atom in atom_list]
        return np.array(positions)
    
    
    def set_position(self,positions):
        #CAUTION: positions should be in the order of atom_list
        center_atom = self.center_atom
        metal_index = self.metal_index
        ligands = self.ligands
        atom_indices_for_each_ligand = self.get_atom_indices_for_each_ligand()
        atom_list = self.get_atom_list()
        n = len(atom_list)
        if len(positions) != n:
            print ("Number of atoms does not match ...")
            return
        
        process.locate_atom(center_atom,positions[self.metal_index])
        for i, atom_indices in enumerate(atom_indices_for_each_ligand):
            ligand = ligands[i]
            for j in range(len(atom_indices)):
                process.locate_atom(ligand.molecule.atom_list[j],positions[atom_indices[j]])
            

    def copy(self):
        geometry_name = self.geometry_type.geometry_name
        center_atom = self.center_atom.copy()
        ligands = [ligand.copy() for ligand in self.ligands]
        chg = self.chg
        multiplicity = self.multiplicity
        atom_indices_for_each_ligand = copy.deepcopy(self.atom_indices_for_each_ligand)
        
        new_complex = MetalComplex(geometry_name,center_atom,ligands,chg,multiplicity)
        new_complex.metal_index = self.metal_index
        new_complex.atom_indices_for_each_ligand = atom_indices_for_each_ligand
        new_complex.name = self.name
        
        return new_complex
        
    
    def get_stereoisomers(self):  
        geometry_type = self.geometry_type
        permutations = geometry_type.permutations
        isomers = []
        if len(permutations) == 0:
            print ("Not supported geometry type ...")
            return isomers
        else:
            for permutation in permutations:
                isomer = self.copy()
                ligands = isomer.ligands
                for i, ligand in enumerate(ligands):
                    for j, binding_info in enumerate(ligand.binding_infos):
                        _, _ = binding_info
                        ligand.binding_infos[j] = (binding_info[0], permutation[i])
                isomers.append(isomer)
        return isomers
            
            
    def get_embedding(self,num_conformer = 10, d_criteria = 0.5, align=True):
        #options = [0, 1, 2]
        options = [0, 1]
        min_d = -0.1 #0.1
        max_d = 0.4
        num_conf_per_option = int((num_conformer+1)/2)
        if num_conf_per_option == 0:
            num_conf_per_option = 1
        if num_conf_per_option > 1:
            scale_size = min(0.1,0.4/(num_conf_per_option-1))
            start = max(0.8,1 - scale_size * (num_conf_per_option-1)/2)
            end = min(1.2,1 + scale_size * (num_conf_per_option-1)/2)
            scales = np.arange(start,end,scale_size)
        else:
            scales = np.array([1.0])
        candidate_positions = []
        for option in options:
            for scale in scales:
                if True:
                    positions = embed.get_embedding(self,scale,option,align=align)
                    if positions is not None:
                        candidate_positions.append(positions)
                        #break
                else:
                    continue

            if len(candidate_positions) == num_conformer:
                break
        if len(candidate_positions) == 0:
            print ("No valid embedding found ...")
            exit()
        return candidate_positions


    def print_coordinate_list(self):
        atom_list = self.get_atom_list()
        n = len(atom_list)
        for atom in atom_list:
            element = atom.get_element()
            coordinate = atom.get_coordinate()
            print_x = f"{coordinate[0]:>12.8f}"
            print_y = f"{coordinate[1]:>12.8f}"
            print_z = f"{coordinate[2]:>12.8f}"
            print(f"{element:<3} {print_x} {print_y} {print_z}")
        print()


    def get_distances_from_center(self):
        """
        get the distances of all atoms from the center atom
        """
        metal_index = self.metal_index
        if metal_index is None:
            print ("Metal index is not determined ...")
            return []
        adj_matrix = self.get_adj_matrix()
        neighbor_list = [-1] * len(adj_matrix)
        atom_set = set([metal_index])
        neighbor_list[metal_index] = 0 
        distance = 1
        while len(atom_set) > 0:
            next_set = set()
            for atom in atom_set:
                for i in range(len(adj_matrix)):
                    if adj_matrix[atom][i] == 1 and neighbor_list[i] == -1:
                        neighbor_list[i] = distance
                        next_set.add(i)
            atom_set = next_set
            distance += 1
        return neighbor_list
    
def get_om_from_modified_smiles(smiles):
    from rdkit import Chem
    import ligand
    
    smiles_list = smiles.split('|')
    
    n = len(smiles_list)
    metal_atom = Chem.MolFromSmiles(smiles_list[0])
    metal_chg = Chem.GetFormalCharge(metal_atom)
    metal_atom = chem.Atom(metal_atom.GetAtomWithIdx(0).GetSymbol())
    z_sum = metal_atom.get_atomic_number()
    ligands = [ligand.get_ligand_from_smiles(smiles_list[i]) for i in range(1,n-1)]
    
    ligand_chg = sum([l.molecule.get_chg() for l in ligands])
    z_sum += sum([np.sum(l.molecule.get_z_list()) for l in ligands])
    
    chg = ligand_chg + metal_chg
    geometry_name = smiles_list[-1]
    multiplicity = (z_sum - chg) % 2 + 1
     
    metal_complex = MetalComplex(geometry_name,metal_atom,ligands,chg,multiplicity)
    metal_complex.metal_index = 0
    metal_complex.multiplicity = multiplicity

    return metal_complex
            

if __name__ == "__main__":
    pass
