from rdkit import Chem

def get_ligand_from_smiles(mapped_smiles):
    import process
    
    params = Chem.SmilesParserParams()
    params.removeHs = False
    rd_mol = Chem.MolFromSmiles(mapped_smiles,params)
    rd_mol = Chem.AddHs(rd_mol)
    ace_mol = process.get_ace_mol_from_rd_mol(rd_mol)
    n = len(ace_mol.atom_list)
    binding_infos = dict()
    for i in range(n):
        rd_atom = rd_mol.GetAtomWithIdx(i)
        mapping = rd_atom.GetAtomMapNum()
        if mapping > 0:
            if mapping in binding_infos:
                binding_infos[mapping].append(i)
            else:
                binding_infos[mapping] = [i]
    final_binding_infos = []
    for position in binding_infos:
        final_binding_infos.append([binding_infos[position],position])
    
    ligand = Ligand(ace_mol,final_binding_infos)
    return ligand

class Ligand:

    def __init__(self,molecule,binding_infos):
        self.molecule = molecule # chem.Molecule
        self.binding_infos = binding_infos # [[[int],int]]        
    
    
    def get_smiles(self):
        return self.molecule.get_smiles('ace')
    

    def get_denticity(self):
        return len(self.binding_infos)    
    

    def update(self):
        self.coord_list = self.molecule.get_coordinate_list()
    
    
    def get_adj_matrix(self):
        adj_matrix = self.molecule.adj_matrix
        if adj_matrix is None:
            adj_matrix = self.molecule.get_adj_matrix()
        return adj_matrix


    def print_coordinate_list(self):
        coordinate_list = self.coordinate_list()
        if coordinate_list is None:
            coordinate_list = self.get_coordinate_list()
        n = len(coordinate_list)
        
        print(n)
        print()
        for i in range(n):
            symbol, x, y, z = self.coordinate_list[i]
            prinx_x = f'{x:.6f}'
            print_y = f'{y:.6f}'
            print_z = f'{z:.6f}'
            print(f'{symbol} {print_x} {print_y} {print_z}')
        print()
        
    def copy(self):
        import copy
        molecule = self.molecule.copy()
        molecule.adj_matrix = self.molecule.get_adj_matrix()
        binding_infos = copy.deepcopy(self.binding_infos)
        new_ligand = Ligand(molecule,binding_infos)
        return new_ligand
                

    def __str__(self):
        pass
        
