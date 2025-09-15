"""
---generate_molecule.py---
Generate 2D graph molecule in ACE-Reaction format from xyz files or SMILES
"""

import os 
import subprocess
import copy
import numpy as np 
import random

from rdkit import Chem
import rdkit.Chem.rdchem as rdchem
from scipy import spatial

from MetalloGen import chem

from MetalloGen.utils import am
from MetalloGen.utils import frag
from MetalloGen.utils import compute_chg_and_bo_pulp as compute_pulp

def compute_bond_order_from_connectivity(molecule,chg,method = 'sum_of_fragments'):
    """ Returns bo_matrix by using heuristics (maxvalency and chg) for given total charge of molecule
    Args:
        |  molecule (<class 'Molecule'>)
        |  chg (int): Charge of the molecule
        |  method (string): method of computing charge, mainly 'sum_of_fragments' and 'ionic', normally we use sum_of_fragments
    Returns:
        |  molecule containing those classes: chg of each atom in atom_list in molecule
    """
    atom_list = molecule.atom_list
    z_list = molecule.get_z_list()
    du_list = []
    list_count = 0
    terminal_oxygen_list = []
    for i in range(len(z_list)):
        terminal_oxygen_list.append([])
    for i in range(len(z_list)):
        atom = atom_list[i]
        a = atom_list[i].element
        valency = atom.get_valency()
        neighbor_idx_list = atom.get_neighbor_idx_list()
        if a=='CO' or a=='RH' or a=='NI' or a=='TI' or a=='PD' or  a=='FE' or a=='HF':
            if valency >= 6: 
                du = 0
            else: 
                du = 6 - valency  #For make M-L multiple bonds possible
        elif a=='I':
            if valency >= 6: 
                du = 0
            else: 
                du = 3 - valency  #For make M-L multiple bonds possible
        elif a == 'P' or a == 'N':
            if a == 'P' and valency == 4: 
                du = [0,1] 
                list_count += 1
            elif a == 'P' and valency == 3: 
                du = [0,2] 
                list_count += 1
            # CN or N2 case
            elif a == 'N' and valency == 1 and (atom_list[neighbor_idx_list[0]].element=='N' or atom_list[neighbor_idx_list[0]].element=='C'): 
                du = [1,2] 
                list_count += 1
            elif valency == 3: 
                if list(map(lambda x:Elemlist[x],[a for a,b in enumerate(list(Adj[i])) if b==1])).count('N')>=1: 
                    du = 0
                else: 
                    du = [0,1] 
                    list_count += 1
            elif valency == 2: 
                du = [1,2] 
                list_count += 1
            elif valency == 1:
                du = [2,3] 
                list_count += 1
            else: 
                du = atom.get_max_valency() - valency
        elif a == 'B':
            du = 3 - valency
        elif a == 'S':
            if valency == 1:
                du = 1
            elif valency == 2:
                du = [0,2]
                list_count += 1
            elif valency == 3:
                du = [1,3]
                list_count += 1
            else:
                du = atom.get_max_valency() - valency
        elif a == 'O' and valency == 1:
            bonding_atom = neighbor_idx_list[0]
            valency_prime = atom_list[bonding_atom].get_valency()
            if str.upper(atom_list[bonding_atom].Type) == 'C' and valency_prime == 1: 
                du = 2 # carbon monoxide
            elif str.upper(atom_list[bonding_atom].Type) == 'N' and valency_prime == 3: 
                terminal_oxygen_list[bonding_atom].append(i) 
                du = 0 # for N-O single bond 
            else: 
                du = 1
        else:
            du = atom_list[i].get_valency() - valency
            if du < 0: 
                du = 0
        du_list.append(du)

    New_Adj=np.copy(Adj)

    for i in range(len(Terminal_oxygens)):
        NObonds=Terminal_oxygens[i]
        if len(NObonds)==2: #NO2 terminal
            oxygen1=NObonds[0]
            New_Adj[i][oxygen1]=New_Adj[oxygen1][i]=2
            list_count-=1; DUs[i]=0

    for i in range(len(Elemlist)):
        a=Elemlist[i]
        if type(DUs[i])==int and  DUs[i]>0:
            neighbor_DUs=[]
            for j in range(len(New_Adj)):
                if New_Adj[i][j]!=0:
                    if type(DUs[j])==list: neighbor_DUs.append(1)
                    else: neighbor_DUs.append(DUs[j])
            if sum(neighbor_DUs)==0: 
                if type(DUs[i])==list: list_count-=1
                DUs[i]=0

    multiple_valence_count=sum(isinstance(i, list) for i in DUs)
    if multiple_valence_count!=list_count: list_count=multiple_valence_count

    if list_count==0 and sum(DUs)<=0:
        if ObtainAllResonances: return [New_Adj]
        return New_Adj #No unsatu atoms

    #Create possibilities for valences
    Candidate_DUs=[DUs]
    if list_count>0:
        while True:
            new_candidate=[]
            for DUs in Candidate_DUs:
                for i in range(len(DUs)):
                    if type(DUs[i])==list:
                        newDU1=DUs[:i]+[DUs[i][0]]+DUs[i+1:]
                        newDU2=DUs[:i]+[DUs[i][1]]+DUs[i+1:]
                new_candidate.append(newDU1); new_candidate.append(newDU2)
            Candidate_DUs=new_candidate
            if len(Candidate_DUs)==2**list_count: break

    ##sort candidate DUs
    for i in range(len(Candidate_DUs)):
        for j in range(i+1,len(Candidate_DUs)):
            sumDUi=sum(Candidate_DUs[i]); sumDUj=sum(Candidate_DUs[j])
            if sumDUi<sumDUj:
                temp=Candidate_DUs[i][:]
                Candidate_DUs[i]=Candidate_DUs[j][:]
                Candidate_DUs[j]=temp[:]

    Candidate_Unsatu_set=[]
    for DUs in Candidate_DUs:
        Unsatu_set=[]
        for i in range(len(DUs)):
            if DUs[i]>0: Unsatu_set.append(i)
        Candidate_Unsatu_set.append(Unsatu_set)

    AllResonances=[]

    Current_BO=np.copy(New_Adj)
    Current_FC=getFC(atom_list,Current_BO,Total_charge,TotalChargeMethod)
    for i in range(len(Candidate_DUs)):
        if len(Candidate_Unsatu_set[i])>0:
            if ObtainAllResonances:
                AllBOs=FindResonances(atom_list,New_Adj,Candidate_Unsatu_set[i],Candidate_DUs[i],Total_charge)
                AllResonances+=AllBOs
            else:
                Candidate_BO=Obtain_Candidate_BOs(atom_list,New_Adj,Candidate_Unsatu_set[i],Candidate_DUs[i],Total_charge)
                Candidate_FC=getFC(atom_list,Candidate_BO,Total_charge,TotalChargeMethod)

                charge_balance=(Total_charge==sum(Candidate_FC))

                if charge_balance: 
                    NumberOfNeutralAtoms=Candidate_FC.count(0)
                    if len(Candidate_FC)-abs(int(Total_charge))==NumberOfNeutralAtoms: return Candidate_BO

                    if NumberOfNeutralAtoms > Current_FC.count(0) or np.sum(Candidate_BO) > np.sum(Current_BO):
                        Current_BO=np.copy(Candidate_BO); Current_FC=Candidate_FC[:]
                elif np.sum(Candidate_BO) > np.sum(Current_BO): 
                    Current_BO=np.copy(Candidate_BO); Current_FC=Candidate_FC[:]

    if not ObtainAllResonances:
        MetalElements=['SC','TI','V' ,'CR','MN','FE','CO','NI','CU','ZN',\
                        'Y','ZR','NB','MO','TC','RU','RH','PD','AG','CD',\
                        'LU','HF','TA','W' ,'RE','OS','IR','PT','AU','HG'] 

        if len(set(Elemlist) & set(MetalElements))!=0: Current_BO=Detect_MetalCarbonyl(Elemlist,Current_BO)

    if ObtainAllResonances: 
        Final_Resonances=[]
        Ceiglist=[]
        for oneBO in AllResonances:
            Resonance_FC=[int(x) for x in getFC(atom_list,oneBO,Total_charge,TotalChargeMethod)]
            if sum(Resonance_FC)!=int(Total_charge): continue
            if 2 in Resonance_FC or -2 in Resonance_FC: continue

            samecheck=False
            for prev_BO in Final_Resonances:
                if np.array_equal(prev_BO,oneBO): samecheck=True; break
            if not samecheck: Final_Resonances.append(oneBO)
        return Final_Resonances
    return Current_BO
      

def compute_chg_from_chg(molecule,chg,bo_candidate = None,method = 'sum_of_fragments'):
    """ Returns chg_list by using heuristics (maxvalency and chg) for given total charge of molecule
    Args:
        |  molecule (<class 'Molecule'>): class instance of Molecule
        |  chg (int): Charge of the molecule
        |  bo_candidate (list of dict): possible candidates of BO, if None, molecule should contain such data
        |  method (string): method of computing charge, mainly 'sum_of_fragments' and 'ionic', normally we use sum_of_fragments
    Returns:
        |  chg_list: chg of each atom in atom_list in molecule
    """
    atom_list = molecule.atom_list
    total_num_electrons = sum([atom.atomic_number for atom in atom_list])-chg
    chg_list = []
    for i in range(len(atom_list)):
        fc = 0
        atom = atom_list[i]
        neighbors_info = atom.neighbors_info
        total_bond_order = 0
        bo_list = []
        if bo_candidate == None:
            for neighbor_atom_idx in neighbors_info:
                total_bond_order += neighbors_info[neighbor_atom_idx]
                bond_order_list.append(neighbors_info[neighbor_atom_idx])
        else:
            # Compute total_bond_order using
            pass 
        if len(atom_list) == 1: 
            fc = chg 
        elif str.upper(atom_list[i].element) == 'H':
            if total_bond_order == 1: 
                fc = 0
            else:
                if len(atom_list) == 1: 
                    fc = chg
                else: 
                    fc = 0
        elif str.upper(atom_list[i].element) == 'NA': 
            fc = 1 - total_bond_order
        elif (str.upper(atom_list[i].element) == 'F' or str.upper(atom_list[i].element) == 'CL') and len(atom_list) == 1: 
            fc = -1
        elif str.upper(atom_list[i].element) == 'S' and total_bond_order >= 4: 
            if total_bond_order == 4 or total_bond_order == 6: 
                fc = 0 #valence expansion
            if total_bond_order == 5: 
                fc = 1 #valence expansion
        elif str.upper(atom_list[i].element) == 'P' and total_bond_order >= 4: 
            fc = 5-total_bond_order  #fc=0 #valence expansion
        elif str.upper(atom_list[i].element) == 'RH' or str.upper(atom_list[i].element) == 'CO' or str.upper(atom_list[i].element) == 'NI' or str.upper(atom_list[i].element) == 'TI' or str.upper(atom_list[i].element) == 'PD' or str.upper(atom_list[i].element) == 'FE': 
            fc = 0 # Transition metals - tricky..
        elif str.upper(atom_list[i].element) == 'C' and bond_order_list.count(1) == 2 and bond_order_list.count(2) == 0: 
            fc = 0 #carbene
        elif str.upper(atom_list[i].element) == 'C' and bond_order_list.count(1) == 0 and bond_order_list.count(2) == 1: 
            fc = 0 #=C terminus
        elif str.upper(atom_list[i].element)=='B':
            if total_bond_order == 0: 
                fc = 0
            else: 
                fc = 3-total_bond_order
        elif total_bond_order == 0: 
            fc = 0 #For a single atom, regarded as zero
        elif str.upper(atom_list[i].element) == 'I' and total_bond_order > 1: #valence shell expansion for I
            if total_bond_order == 3 or total_bond_order == 5: 
                fc = 0
            if total_bond_order == 2 or total_bond_order == 4: 
                fc = 1
        else:
            lonepair = (4-total_bond_order)*2
            fc = atom_list[i].group - total_bond_order - lonepair
            
        chg_list.append(int(fc))

    for i in range(len(atom_list)):
        if str.upper(atom_list[i].element) == 'C' and bond_order_list.count(1) == 3: #Three single bonds. carbocation or carbanion
            if chg > 0: 
                chg_list[i] = 1
            elif chg < 0: 
                chg_list[i] = -1
            else: 
                chg_list[i] = 0

    if method == 'sum_of_fragments' and total_num_electrons%2 != 0 and sum(chg_list) != chg:
        for i in range(len(chg_list)):
            if chg_list[i] != 0:  
                chg_list[i] = 0 #RADICAL
            if sum(chg_list) == chg: 
                break

    #Exceptional: Carbonyl monooxide
    if len(atom_list) == 2 and sorted([str.upper(atom_list[0].element),str.upper(atom_list[1].element)]) == ['C','O'] and total_bond_order == 3:
        chg_list = [0,0]
    return chg_list

def get_optimal_molecule(ace_molecule,chg,unsaturated_atom_idx_list = None,unsaturation_degree_list = None):
    """ Returns the best bo connectivity of molecule for given adjacency
    Args:
        |  ace_molecule (<class 'Molecule'>): class instance of Molecule
        |  chg (int): total chg of molecule
        |  unsaturated_atom_idx_list (list of integer): indices of unsaturated atoms
        |  unsaturation_degree_list (list of integer): number of degree of unsaturated atoms
    Returns:
        |  optimal_molecule (<class 'Molecule'>): class instance of Molecule that has maximal bond (total bond order sum is the maximum for given adjacency)  
    """
    atom_list = ace_molecule.atom_list
    n = len(atom_list)
    total_adj_sum = 0
    neighbors_info_list_with_adj = []
    optimal_neighbors_info_list = []
    if unsaturated_atom_idx_list == None:
        print ('We will fix here...')
    # First obtain sum of adj and neighbor_info_with_adj
    for i in range(n):
        neighbors_info = atom_list[i].neighbors_info
        if neighbors_info == None:
            print ('Neighbor information is not updated!!!')
        else:
            new_neighbor_dict = dict()
            for neighbor_atom_idx in neighbors_info:
                new_neighbor_dict[neighbor_atom_idx] = 1
                total_adj_sum += 1
            neighbors_info_list_with_adj.append(new_neighbor_dict)
    # Now, find new bonds
    maximum_formed_bond = -10000
    maximum_formed_bond_set = []
    for start_point in range(len(unsaturated_atom_idx_list)):
        new_formed_bond = 0
        additional_bonds = []
        search_atom_idx_list = unsaturated_atom_idx_list[:]
        search_degree_list = unsaturation_degree_list[:]
        sum_of_degree = sum(search_degree_list)
        search_atom_idx_list.insert(0,search_atom_idx_list.pop(start_point))
        current_atom = search_atom_idx_list[0]
        while search_atom_idx_list!=[]:
            first_neighbors = atom_list[current_atom].get_neighbor_idx_list()
            if len(neighbors) == 0:
                search_atom_idx_list.pop(search_atom_idx_list.index(current_atom))
            else:
                second_num_neighbors = []
                for first_neighbor in first_neighbors:
                    second_num_neighbors.append(atom_list[first_neighbor].neighbors_info)
                target = first_neighbors[second_num_neighbors.index(min(second_num_neighbors))]
                additional_bond_order = min(search_degree_list[current_atom],search_degree_list[target])
                additional_bonds.append([current_atom,target,additional_bond_order])
                additional_bonds.append([target,current_atom,additional_bond_order])
                search_degree_list[target] -= additional_bond_order
                search_degree_list[current_atom] -= additional_bond_order
                if search_degree_list[current_atom] <= 0:
                    search_atom_idx_list.pop(search_atom_idx_list.index(current_atom))
                if search_degree_list[target] <= 0:
                    search_atom_idx_list.pop(search_atom_idx_list.index(target))
            if search_atom_idx_list == []:
                break
            remaining_atoms_neighbors = []
            for remaining_atom in search_atom_idx_list:
                remaining_atoms_neighbors.append(len(atom_list[remaining_atom].neighbors_info))
            minimum_neighbor_index = remaining_atoms_neighbors.index(min(remaining_atoms_neighbors))
            current_atom = search_atom_idx_list[minimum_neighbor_index]
        # Now obtain candidate
        if new_formed_bond == sum_of_degree:
            return additional_bonds, new_formed_bond
        if new_formed_bond > maximum_formed_bond:
            maximum_formed_bond = new_formed_bond
            maximum_formed_set = additional_bonds
    return maximum_formed_bond_set, maximum_formed_bond
    

def get_block_diagonal_adj_from_fragments(fragment_adj_list,total_num_atom = 0):
    """ It returns a single block diagonal matrix by concatenating fragment adjacency matrices
    Args:
        |  fragment_adj_list (list): list of adjacency matrices of input fragments
        |  total_num_atom : number of total_atoms (int)
    Returns:
        |  block_diagonal_matrix (<numpy 'np.array'>): total_atom * total_atom matrix
    """
    if total_num_atom == 0:
        for fragment_adj in fragment_adj_list:
            total_num_atom += len(fragment_adj)
    total_diagonal_matrix = np.zeros((total_num_atom,total_num_atom))
    cnt = 0
    for i in range(len(fragment_adj_list)):
        fragment_adj = fragment_adj_list[i]
        size = len(fragment_adj)
        for j in range(size):
            for k in range(size):
                total_diagonal_matrix[cnt+j][cnt+k] = fragment_adj[j][k]
        cnt += size
    return total_diagonal_matrix    

def locate_molecule(ace_molecule,coordinate_list,update = False):
    """ Locates atoms according to coordinate_list, be cautious on ordering of atoms
    Args:
        |  ace_molecule (<class 'Molecule'>): class instance of Molecule
        |  coordinate_list (list of list (x,y,z)): list of 3d coordinate, where each x,y,z are float type
    Returns:
        |  No return, it direcly modifies the geometry of a given molecule
    """
    atom_list = ace_molecule.atom_list
    for i in range(len(atom_list)):
        atom = atom_list[i]
        atom.set_coordinate(coordinate_list[i])
    if update:
        ace_molecule.set_adj_matrix(None)

def translate_molecule(ace_molecule,vector):
    atom_list = ace_molecule.atom_list
    for atom in atom_list:
        translate_atom(atom,vector)


def locate_atom(atom,coordinate):
    """ Locates a single atom to input 'coordinate'
    Args:
        |  atom (<class 'Atom'>): class instance of Atom
        |  coordinate (list of float): coordinate in form of [x,y,z]
    Returns:
        |  No return, it directly locates the atom to the given coordinate
    """
    atom.x = coordinate[0]
    atom.y = coordinate[1]
    atom.z = coordinate[2]

def translate_atom(atom,vector):
    atom.x += vector[0]
    atom.y += vector[1]
    atom.z += vector[2]


def read_molecule(f,extension='xyz'):
    molecule = chem.Molecule()
    atom_list = []
    info = []
    if extension == 'com':
        try:
            chg,multiplicity = f.readline().strip().split()
            chg = int(chg)
            multiplicity = int(multiplicity)
            molecule.chg = chg
            molecule.multiplicity = multiplicity
        except:
            print ('Wrong format ! Should start with molecular charge and multiplicity !!!')
            return molecule, info
        while True:
            try:
                atom_line = f.readline().strip().split()
                print (atom_line)
                element = atom_line[0]
                x = float(atom_line[1])
                y = float(atom_line[2])
                z = float(atom_line[3])
                atom = chem.Atom(element)
                atom.x = x
                atom.y = y
                atom.z = z
                atom_list.append(atom)
            except:
                break
        molecule.atom_list = atom_list
        return molecule, []

    elif extension == 'xyz':
        try:
            atom_num = int(f.readline().strip())
            info = f.readline().strip().split()
        except:
            atom_num = 0
        for i in range(atom_num):
            atom_line = f.readline().strip().split()
            element = atom_line[0]
            x = float(atom_line[1])
            y = float(atom_line[2])
            z = float(atom_line[3])
            atom = chem.Atom(element)
            atom.x = x
            atom.y = y
            atom.z = z
            atom_list.append(atom)
        molecule.atom_list = atom_list
        return molecule, info

def get_rd_mol_from_np_array(z_list,chg_list,bo_matrix):
    from rdkit import Chem
    n = len(chg_list)
    bond_types = {1:Chem.BondType.SINGLE, 2:Chem.BondType.DOUBLE, 3:Chem.BondType.TRIPLE}
    # First generate atom_list
    rd_mol = Chem.Mol()
    rde_mol = Chem.EditableMol(rd_mol)
    for i in range(n):
        rd_atom = Chem.Atom(int(z_list[i]))
        rd_atom.SetFormalCharge(int(chg_list[i]))
        rde_mol.AddAtom(rd_atom)
    # Get bond list from np.array bo_matrix (Never use), you can expand this part
    bond_order_list = [1,2,3]
    for bond_order in bond_order_list:
        np_array_list = np.where(bo_matrix == bond_order)
        np_array_list = np.stack(np_array_list,axis=1)
        for array in np_array_list:
            begin = array[0]
            end = array[1]
            if begin < end:
                rde_mol.AddBond(int(begin),int(end),bond_types[bond_order])
    rd_mol = rde_mol.GetMol()
    return rd_mol

def get_ace_mol_with_coordinate(smiles):
    molecule = chem.Molecule(smiles)
    coordinate_list = molecule.make_3d_coordinate()
    locate_molecule(molecule,coordinate_list)
    return molecule


def get_ace_mol_from_rd_mol(rd_molecule,add_hydrogen = True,include_stereo = False):
    """ It converts rd_molecule type info ace_molecule type
    Args:
        |  rd_molecule (<class 'rdkit.Molecule>')
    Returns:
        |  ace_molecule(<class Molecule>)
    """
    # Kekulize molecule
    from rdkit import Chem
    rd_molecule_copy = copy.deepcopy(rd_molecule)
    try:
        if add_hydrogen:
            rd_molecule = Chem.AddHs(rd_molecule) 
        Chem.rdmolops.Kekulize(rd_molecule)
    except:
        rd_molecule = rd_molecule_copy
    bond_types = {Chem.BondType.SINGLE:1, Chem.BondType.DOUBLE:2, Chem.BondType.TRIPLE:3} 
    n = rd_molecule.GetNumAtoms()
    atom_list = []
    chg_list = []
    atom_feature = dict()
    # Make atom_list
    for i in range(n):
        rd_atom = rd_molecule.GetAtomWithIdx(i)
        ace_atom = chem.Atom()
        chg_list.append(rd_atom.GetFormalCharge())
        '''
        position = rd_molecule.GetAtomPosition(i)
        if position!=None:
            ace_atom.x = position[0]
            ace_atom.y = position[1]
            ace_atom.z = position[2]
        '''
        ace_atom.atomic_number = rd_atom.GetAtomicNum()
        atom_list.append(ace_atom)
    atom_feature['chg'] = np.array(chg_list)
    # Make bond order matrix
    bonds = rd_molecule.GetBonds()
    bo_matrix = np.zeros((n,n))
    for bond in bonds:
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_order = bond_types[bond.GetBondType()]
        bo_matrix[begin][end] = bo_matrix[end][begin] = bond_order
    ace_molecule = chem.Molecule()
    ace_molecule.atom_list = atom_list
    ace_molecule.bo_matrix = bo_matrix
    ace_molecule.atom_feature = atom_feature
    return ace_molecule

    
def get_total_atom_list_from_molecule_list(molecule_list):
    """ Returns total_atom_list by merging atom_lists in molecule_list, indexing is also changed, 
    Args:
        |  molecule_list (list of class 'Molecule): list of molecules
    Returns:
        |  atom_list (list of class 'Atom'): list of class 'Atom' that contains all atoms in molecule_list 
    """
    atom_list = []
    cnt = 0
    for molecule in molecule_list:
        adding_atom_list = molecule.atom_list
        for atom in adding_atom_list:
            neighbors_info = atom.neighbors_info
            if neighbors_info == None:
                print ('neighbors are not prepared!!! IM gneneration...')
            else:
                new_neighbors_info = dict()
                for neighbor_atom_idx in neighbors_info:
                    new_neighbors_info[neighbor_atom_idx+cnt] = neighbors_info[neighbor_atom_idx]
                neighbors_info = new_neighbors_info
        atom_list += adding_atom_list
        cnt += len(adding_atom_list) 
    return atom_list

def get_block_diagonal_adj_by_distance(molecule_list,coeff = 1.10):
    """ From molecule_list, generate adjacency as block diagonal matrix by appending molecules
    Args: 
        |  molecule_list: list of class Molecule
        |  coeff (float): scaling criteria of recognizing bond
    Returns:
        |  adj (class 'numpy.ndarray') of molecule lists 
    """
    total_atom = 0
    block_len = []
    for molecule in molecule_list:
        total_atom += len(molecule.atom_list)
        block_len.append(len(molecule.atom_list))
    block_diagonal_adj = np.zeros((total_atom,total_atom))
    index =0 
    for molecule in molecule_list:
        adj = get_adj_matrix_from_distance(molecule,coeff)
        block_diagonal_adj[index:index+len(adj)][index:index+len(adj)] = adj[:][:]
        index += len(adj)
    return block_diagonal_adj,block_len
    
def get_condensed_adj(molecule_list):
    """ From molecule list, it returns adjacency of block diagonal matrix without considering hydrogen atoms
    Args: 
        |  molecule_list: list of class Molecule
        |  coeff (float): scaling criteria of recognizing bond
    Returns:
        |  adj (class 'numpy.ndarray') of molecule lists 
    """

def get_kth_neighbor_atom_list(atom_list,atom_center_list,radius = 1,total_set = True):
    n = len(atom_list)
    radius_neighbor_dict = dict()
    current_distance = 1
    total_atom_set = set(atom_center_list)
    # Append center itself, radius 0
    for atom_center in atom_center_list:
        radius_neighbor_dict[atom_center] = [[atom_center]]
    # Next, append neighbors until kth order
    while current_distance <= radius:
        for atom_center in radius_neighbor_dict: 
            neighbor_list_by_radius = radius_neighbor_dict[atom_center]
            checking_atom_idx_list = neighbor_list_by_radius[-1]
            previous_atom_idx_list = []
            m = len(radius_neighbor_dict)
            for i in range(m-1):
                previous_atom_idx_list += radius_neighbor_dict[i]
            next_atoms = []
            # atom with distance current_distance - 1
            for atom_idx in checking_atom_idx_list:
                neighbors_info = atom_list[atom_idx].neighbors_info
                # If neighbor is not in previous one, those are candidate for next atoms
                for neighbor_atom_idx in neighbors_info:
                    if atom_idx not in previous_atom_idx_list:
                        new_atoms.append(neighbor_atom_idx)
                        total_atom_set.add(neighbor_atom_idx)
            # Append atom that is in current_distance
            neighbor_list_by_radius.append(next_atoms)
        current_distance += 1
    if total_set:
        list(total_atom_set) 
    else:
        return radius_neighbor_dict                

def get_kth_neighbor_bond_list(atom_list,atom_center_list,radius = 1,contain_bond_order = False):
    """ Here, we do not put total_set. It returns all bonds whose start and end are atoms within kth_neighbor with respect to atom_center_list
    Args:
        |  atom_list (list of class 'Atom'): list of class 'Atom' 
        |  atom_center_list (int): index list of atom_centers for checking radius
        |  radius (int): positive integer for checking radius-th nearest neighbors
        |  contain_bond_order (boolean): if True, it returns bond with bond_order (i,j,bond_order)
                                       if False, it just returns atom indices of the bond (i,j)
    Returns:
        |  kth_neighbor_bond_list (list of pair (i,j)): where i, j are atom_idx of the two bonds
    """
    kth_neighbor_bond_list = []
    total_neighbor_set = get_kth_neighbor_list(atom_center_list,radius = radius,total_set = True)
    for atom_idx in total_neighbor_set:
        neighbors_info = atom_list[atom_idx].neighbors_info
        for neighbor_atom_idx in neighbors_info:
            if atom_idx<neighbor_atom_idx:
                if contain_bond_order:
                    kth_neighor_bond_list.append((atom_idx,neighbor_atom_idx,neighbors_info[neighbor_atom_idx]))
                else:
                    kth_neighbor_bond_list.append((atom_idx,neighbor_atom_idx)) 
    return kth_neighbor_bond_list



######### Not need to be ready... Just directly frag.py
def get_unsaturated_atom_list(frag_adj_list,atom_list,reactive_atoms,user_defined_valency_list):
    """ Returns informations on active atoms.
    Args:
        |  frag_adj_list (list of np.ndarray): list of adjacency matrices of input fragments
        |  atom_list (list of class Atom): list of atom elements of all fragments (intermediate)
        |  reactive_atoms: list of 
        |  user_defined_valency_list(list of int): list of maximum valences of elements defined by user
           (e.g. ['N',4,'O',3,...])
    Returns:
        |  unsaturated_atom_list (list of int): list of indices of active atoms
        |  unsaturation_degree_list (list of int): list of degress of unsaturation of active atoms (possible number of reaction)
        |  frag_idx_list (list of int): list of indices of fragments to which active atoms belong
        |  z_list (list of int): list of atomic numbers of active atoms
    """
    max_valency = dict()
    if len(user_defined_valency_list)>0:
        for i in range(0,len(user_defined_valency_list),2):
            try:
                max_valency[str.upper(user_defined_valency_list[i])] = user_defined_valency_list[i+1]
            except TypeError:
                max_valency[user_defined_valency_list[i]-1] = user_defined_valency_list[i+1]
    else:
        pass        
    

'''
def append_neighbor(atom,bonded_atom_idx,bond_order):
    """ Modifies add new neighbor of class atom
    Args:
        |  atom (<class 'Atom'>): class instance of atom
        |  bonded_atom_idx(int): The index of a new atom that 'atom' is bonded to
        |  bond_order(int): bond_order
    Returns:
        | No return! It modifies atom!
    """
    if atom.neighbors_info == None:
        atom.neighbors_info = dict()
        atom.neighbors_info[bonded_atom_idx] = bond_order
    else:
        atom.neighbors_info[bonded_atom_idx] = bond_order

'''
def remove_neighbor(atom,bonded_atom_idx,bond_order):
    """ Modifies add new neighbor of class atom
    Args:
        |  atom (<class 'Atom'>): class instance of atom
        |  bonded_atom_idx(int): The index of a new atom that 'atom' is removed
        |  bond_order(int): bond_order
    Returns:
        | No return! It modifies atom!
    """
    if atom.neighbors_info == None:
        print ('Wrong input!')
    else:
        del(atom.neighbors_info[bonded_atom_idx])


########### From here, we have some operations that deal with multiple objects: intersection, different set, union,
 
def get_desired_set_from_two_molecule_list(molecule_list1,molecule_list2,desired_set):
    """ Returns obtainable sets on molecule_lists
    Args:
        |  molecule_list1, molecule_list2: two different molecule lists
        |  type_of_set (list of string): type of set the user want to get. Mainly 3 possible sets are obtainable ("union", "difference", "intersection")
    Returns:
        |  set_list (dict of list): dict with 4 keys: "union", "1-2", "2-1", "intersection"
                                  : set_list["string"] is the list of class "Molecule" resulted by two molecule_lists with operation "string"
    """
    set_list = dict()
    n1 = len(molecule_list1)
    n2 = len(molecule_list2)
    molecule_idx_list1 = set(range(n1))
    molecule_idx_list2 = set(range(n2))
    difference_list1 = None # 1-2
    difference_list2 = None # 2-1
    intersection_list = [set(),set(),] # Index for molecule_list1, molecule_list2 respectively 
    # For every loop, obtain those functions
    for i in molecule_idx_list1:
        molecule1 = molecule_list1[i]
        for j in molecule_idx_list2:
            molecule2 = molecule_list2[j]
            if molecule1.is_same_molecule(molecule2):
                intersection_list[0].add(i)
                intersection_list[1].add(j)
    difference_set1 = molecule_idx_list1 - intersection_list[0] # A - (A&B)
    difference_set2 = molecule_idx_list2 - intersection_list[1] # B - (A&B)
    molecule_intersection_list = list(map(lambda x:molecule_list1[x],intersection_list[0]))
    molecule_union_list = list(map(lambda x:molecule_list1[x],molecule_idx_list1))+list(map(lambda x:molecule_list2[x],difference_set2))  # A|B = A | (B-A)
    molecule_difference_list1 = list(map(lambda x:molecule_list1[x],difference_set1))
    molecule_difference_list2 = list(map(lambda x:molecule_list2[x],difference_set2))
    for desired_type in desired_set:
        if desired_type == 'union':
            set_list[desired_type] = molecule_union_list 
        elif desired_type == 'difference':
            set_list['1-2'] = molecule_difference_list1
            set_list['2-1'] = molecule_difference_list2
            
        elif desired_type == 'intersection':
            set_list[desired_type] = molecule_intersection_list
    return set_list

def get_atom_list_from_z_list(z_list):
    atom_list = []
    for atomic_number in z_list:
        atom = chem.Atom(atomic_number)
        atom_list.append(atom)
    return atom_list
            
def get_atom_list_from_element_list(element_list):
    atom_list = []
    for element in element_list:
        atom = chem.Atom(element)
        atom_list.append(atom)
    return atom_list

def get_z_list_from_atom_list(atom_list):
    z_list = []
    for atom in atom_list:
        z_list.append(atom.get_atomic_number())
    return z_list
            
def get_element_list_from_atom_list(atom_list):
    element_list = []
    for atom in atom_list:
        element_list.append(atom.get_element())
    return element_list


def copy_atom_list(atom_list,include_geometry = False):
    new_atom_list = []
    for atom in atom_list:
        new_atom = copy.deepcopy(atom)
        new_atom_list.append(new_atom)
    return new_atom_list

def copy_molecule(molecule,shallow = True, include_geometry = False):
    new_molecule = chem.Molecule()
    if molecule.bo_matrix is not None:
        new_molecule.bo_matrix = np.copy(molecule.bo_matrix)
        new_molecule.adj_matrix = np.copy(molecule.adj_matrix)
    elif molecule.adj_matrix is not None:
        molecule.adj_matrix = np.copy(molecule.adj_matrix)
    new_molecule.chg = molecule.chg
    if shallow:
        if 'chg' in molecule.atom_feature:
            new_molecule.atom_feature['chg'] = copy.deepcopy(molecule.atom_feature['chg'])
    else:
        new_molecule.atom_feature = copy.deepcopy(molecule.atom_feature)
    new_molecule.atom_list = copy_atom_list(molecule.atom_list,include_geometry)
    return new_molecule
            
def copy_molecule_list(molecule_list,include_geometry = False):
    new_molecule_list = []
    for molecule in molecule_list:
        new_molecule_list.append(copy_molecule(molecule))
    return new_molecule_list

def copy_intermediate(intermediate,include_geometry = False):
    atom_list = intermediate.atom_list
    adj_matrix = intermediate.get_matrix('adj')
    bo_matrix = intermediate.get_matrix('bo')
    chg_list = intermediate.get_chg_list()
    new_intermediate = chem.Intermediate((atom_list,adj_matrix,bo_matrix,chg_list))
    return new_intermediate 

def compare_atom_list(atom_list1,atom_list2):
    n = len(atom_list1)
    m = len(atom_list2)
    element_list1 = []
    element_list2 = []
    if n!=m:
        return False          
    else:
        for i in range(n):
            element_list1.append(atom_list1[i].get_element())
            element_list2.append(atom_list2[i].get_element())
            if not atom_list1[i].is_same_atom(atom_list2[i]):
                return False
        return True

def compare_molecule_list(molecule_list1,molecule_list2):
    n = len(molecule_list1)
    m = len(molecule_list2)
    if n != m:
        return False
    for i in range(n):
        molecule1 = molecule_list1[i]
        appeared = False
        for j in range(m):
            molecule2 = molecule_list2[j]
            if molecule1.is_same_molecule(molecule2):
                appeared = True
                break
        if not appeared:
            return False
    return True

def get_molecule_list_without_repitition(molecule_list):
    new_molecule_list = []
    for molecule in molecule_list:
        new_molecule = True
        for molecule_prime in new_molecule_list:
            if molecule.is_same_molecule(molecule_prime):
                new_molecule = False
        if new_molecule:
            new_molecule_list.append(molecule)
    return new_molecule_list

def get_permuted_molecule(molecule,permutation):
    atom_list = molecule.atom_list
    bo_matrix = molecule.bo_matrix
    adj_matrix = molecule.adj_matrix
    atom_feature = molecule.atom_feature
    new_molecule = chem.Molecule()
    if bo_matrix is not None:
        bo_matrix = get_permuted_matrix(bo_matrix,permutation)
        adj_matrix = np.where(bo_matrix>0,1,0)
    elif adj_matrix is not None:
        adj_matrix = get_permuted_matrix(adj_matrix,permutation)
    new_molecule.adj_matrix = adj_matrix
    new_molecule.bo_matrix = bo_matrix
    new_molecule.atom_list = get_permuted_atom_list(atom_list,permutation)
    new_molecule.atom_feature = get_permuted_atom_feature(atom_feature,permutation)
    return new_molecule

def get_permuted_atom_list(atom_list,permutation):
    n = len(permutation)
    new_atom_list = [None] * n
    for i in range(n):
        new_atom_list[permutation[i]] = atom_list[i]
    return new_atom_list

def get_permuted_atom_feature(atom_feature,permutation):
    new_atom_feature = dict()
    n = len(permutation)
    if atom_feature == None:
        return atom_feature
    for feature in atom_feature:
        feature_value = atom_feature[feature]
        new_feature_value = copy.deepcopy(feature_value)
        for i in range(n):
            value = permutation[i]
            new_feature_value[value] = feature_value[i]
        new_atom_feature[feature] = new_feature_value
    return new_atom_feature
     
def get_permuted_matrix(matrix,permutation):
    n = len(matrix)
    permuted_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            permuted_matrix[permutation[i]][permutation[j]] = matrix[i][j]
    return permuted_matrix

def get_adj_matrix_from_distance(molecule,coeff = 1.10,criteria=1.0):
    """
    Returns adj_matrix from 3d coordinate of given molecule
    It recognizes bond between two atoms, if the sum of radius * coeff is less than distance between two atoms

    :param coeff(float):
        criteria for recognizing bond. If criteria gets higher, more and more bonds are generated between atoms, 
        since criteria distance for bond distance gets higher.
        Appropriate criteria value is between 0.8 ~ 1.3, here we set default value as 1.10
    
    :return adj(pyclass 'numpy.ndarray'):
        connectivity matrix between atoms
         
    """
    MetalElements=['Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni','Cu','Zn',\
                        'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',\
                        'Lu','Hf','Ta','W' ,'Re','Os','Ir','Pt','Au','Hg'] 
    
    atom_list = molecule.atom_list
    n = len(atom_list)
    radius_list = molecule.get_radius_list()
    radius_matrix_flatten = np.repeat(radius_list,n)
    radius_matrix = radius_matrix_flatten.reshape((n,n))
    radius_sum_matrix = radius_matrix + radius_matrix.T
    coordinate_list = molecule.get_coordinate_list()
    # print(coordinate_list)
    distance_matrix = spatial.distance_matrix(coordinate_list,coordinate_list)
    ratio_matrix = distance_matrix/radius_sum_matrix
    adj1 = np.where(distance_matrix<criteria,1,0)
    adj2 = np.where(ratio_matrix<coeff,1,0)
    adj3 = np.zeros((n,n))
    for i in range(n):
        atom = atom_list[i]
        a = atom.element
        if a in MetalElements:
            dist = radius_list[i]
            max_dist = radius_list[i]+1.0
            for j in range(i+1,n):
                b = atom_list[j].element
                if b != 'H' and distance_matrix[i][j] < max_dist:
                    adj3[i][j] = adj3[j][i] = 1
                elif b == 'H' and distance_matrix[i][j] < dist+0.4:
                    adj3[i][j] = adj3[j][i] = 1
    
    adj = np.where(adj1+adj2+adj3>0,1,0)
    #adj = np.where(adj1+adj2>0,1,0)
    #adj = adj2
    np.fill_diagonal(adj,0)
    return adj 

def get_bo_matrix_from_adj_matrix(molecule,chg=None,method='SumofFragments',obtain_all_resonance = False):
    """
    Returns bo_matrix from adj_matrix stored in pyclass 'Molecule'
    
    :param chg(int):
        total charge of the molecule

    :param method(str):
        Mainly 'SumofFragments' and 'Ionic' is possible
        'SumofFragments' use user defined fragment charge to evaluate bo_matrix 
        (also uses some chemical heuristics)
        'Ionic' method uses chemical heuristics to evaluate bo_matrix

    :param obtain_all_resonance(boolean):
        If True, it returns multiple possible resonance structures, therefore return as list of numpy.ndarray
        (Normally this function is not used)

    :return bo_matrix(pyclass 'numpy.ndarray' or list of pyclass 'numpy.ndarray'):
        possible bo_matrices for obtain_all_resonance=True, otherwise just single bo_matrix
    """
    atom_list = molecule.atom_list
    adj_matrix = molecule.get_adj_matrix()
    if chg is None:
        chg = molecule.get_chg()
        if chg is None:
            print ('Total charge is not specified! Provide charge information!')
            return None
    if obtain_all_resonance:
        bo_candidates = frag.AdjtoBO(atom_list,adj_matrix,chg,method,True)
        return bo_candidates
    else:
        bo_matrix = frag.AdjtoBO(atom_list,adj_matrix,chg,method,False)
        return bo_matrix

def get_chg_list_from_bo_matrix(molecule,chg,bo_matrix,method = 'SumofFragments'):
    """
    Returns chg_list from a given bo_matrix stored in pyclass 'Molecule'
    
    :param chg(int):
        total charge of the molecule

    :param bo_matrix(pyclass 'numpy.ndarray'):
        Possible bond order matrix of given molecule

    :param method(str):
        Mainly 'SumofFragments' and 'Ionic' is possible
        'SumofFragments' use user defined fragment charge to evaluate bo_matrix 
        (also uses some chemical heuristics)
        'Ionic' method uses chemical heuristics to evaluate bo_matrix

    :return chg_list(pyclass 'numpy.ndarray' (1D-array)):
        formal charge of each atom
    """
    atom_list = molecule.atom_list
    #bo_matrix_before = np.copy(bo_matrix)
    chg_list = frag.getFC(atom_list,bo_matrix,chg,method)
    return np.array(chg_list)

def get_chg_list_and_bo_matrix_from_adj_matrix(
    molecule, chg=None, method="SumofFragments"
):
    atom_list = molecule.atom_list
    adj_matrix = molecule.get_adj_matrix()
    if chg is None:
        chg = molecule.get_chg()
        if chg is None:
            print("Total charge is not specified! Provide charge information!")
            return None, None
    bo_matrix = get_bo_matrix_from_adj_matrix(molecule, chg, method)
    chg_list = get_chg_list_from_bo_matrix(molecule, chg, bo_matrix, method)
    return chg_list, bo_matrix

def get_chg_and_bo(molecule,chg=None,method='SumofFragments'):
    atom_list = molecule.atom_list
    adj_matrix = molecule.get_adj_matrix()
    #print (method,chg)
    if chg is None:
            chg = molecule.get_chg()
            if chg is None:
                print ('Total charge is not specified! Provide charge information!')
                return None,None
#    if method == 'gurobi':
#        chg_list, bo_matrix = compute_gurobi.get_chg_and_bo(molecule, chg)
    elif method == 'scipy':
        chg_list, bo_matrix = compute_scipy.get_chg_and_bo(molecule,chg)
    else:
        bo_matrix = get_bo_matrix_from_adj_matrix(molecule,chg,method)
        chg_list = get_chg_list_from_bo_matrix(molecule,chg,bo_matrix,method)
    return chg_list,bo_matrix


def get_chg_list_and_bo_matrix_pulp(molecule, chg=None, **kwargs):
    return compute_pulp.compute_chg_and_bo(molecule, chg, resolve=True, cleanUp=True,HalogenConstraint=True, **kwargs)
    

def get_reduced_intermediate(intermediate,reduce_function):
    reduced_intermediate = chem.Intermediate()
    adj_matrix = intermediate.get_adj_matrix()
    bo_matrix = intermediate.get_bo_matrix()
    chg_list = intermediate.get_chg_list()
    n = len(reduce_function)
    if type(reduce_function) is list:
        reduced_atom_list = []
        index_function = np.ix_(reduce_function,reduce_function)
        reduced_chg_list = []
        reduced_atom_list = [intermediate.atom_list[index] for index in reduce_function]
        reduced_adj_matrix = adj_matrix[index_function]
        if reduced_bo_matrix is not None:
            reduced_bo_matrix = bo_matrix[index_function]
        else:
            reduced_bo_matrix = None
        if chg_list is not None:
            reduce_chg_list = np.array([chg_list[index] for index in reduce_function]) 
    else:
        reduced_atom_list = [None] * n
        reduced_adj_matrix = np.zeros((n,n))
        if bo_matrix is not None:
            reduced_bo_matrix = np.zeros((n,n))
        else:
            reduced_bo_matrix = None
        reduced_chg_list = []
        if chg_list is not None:
            reduced_chg_list = np.zeros((n))
        for original_i in reduce_function:
            i = reduce_function[original_i]
            reduced_atom_list[i] = intermediate.atom_list[original_i]
            if chg_list is not None:
                reduced_chg_list[i] = chg_list[original_i]
            for original_j in reduce_function:
                j = reduce_function[original_j]
                reduced_adj_matrix[i][j] = adj_matrix[original_i][original_j]
                if reduced_bo_matrix is not None:
                    reduced_bo_matrix[i][j] = bo_matrix[original_i][original_j]

    reduced_intermediate.atom_list = reduced_atom_list
    reduced_intermediate.adj_matrix = reduced_adj_matrix
    reduced_intermediate.bo_matrix = reduced_bo_matrix
    if len(reduced_chg_list) > 0:
        reduced_intermediate.atom_feature['chg'] = reduced_chg_list
    return reduced_intermediate

def get_ace_mol_from_minimal_data(minimal_data,object_type = 'molecule'): 
    z_list = minimal_data['z']
    adj_matrix = minimal_data['adj']
    bo_matrix = minimal_data['bo']
    chg = minimal_data['chg']
    chg_list = minimal_data['atom chg']
    coordinate_list = minimal_data['coords']
    if object_type == 'molecule':
        molecule = chem.Molecule((z_list,adj_matrix,bo_matrix,chg_list))
    else:
        molecule = chem.Intermediate((z_list,adj_matrix,bo_matrix,chg_list))
    if chg is not None:
        molecule.chg = chg
    if coordinate_list is not None and coordinate_list[0][0] is not None: # x value of zeroth atom
        locate_molecule(molecule,coordinate_list)
    return molecule

def add_atoms(ace_mol,new_atom_list):
    atom_list = ace_mol.atom_list
    atom_feature = ace_mol.atom_feature
    new_chg_list = np.zeros((len(new_atom_list)))
    n = len(atom_list)
    m = len(new_atom_list)
    bo_matrix = ace_mol.get_matrix('bo')
    if bo_matrix is None:
        adj_matrix = ace_mol.get_matrix('adj')
        if adj_matrix is None:
            print ('Cannot add atoms, since adjacency is not given')
            exit()
        new_adj_matrix = np.zeros((n+m,n+m))
        new_adj_matrix[:n,:n] = adj_matrix
        ace_mol.adj_matrix = new_adj_matrix
    else:
        bo_matrix = ace_mol.get_matrix('bo')
        new_bo_matrix = np.zeros((n+m,n+m))
        new_bo_matrix[:n,:n] = bo_matrix
        ace_mol.bo_matrix = new_bo_matrix   
    for key in atom_feature:
        if key != 'chg':
            atom_feature[key] = None
        else:
            atom_feature[key] = np.concatenate((atom_feature[key],new_chg_list),axis=0)
    for atom_type in new_atom_list:
        new_atom = chem.Atom(atom_type)
        atom_list.append(new_atom)
        
def add_bonds(ace_mol,bond_list):
    bo_matrix = ace_mol.get_matrix('bo')
    if bo_matrix is None:
        print ('impossible to add bond, since bond order is not given!')
        exit()
    else:
        for bond in bond_list:
            start = bond[0]
            end = bond[1]
            bond_order = bond[2]
            bo_matrix[start][end] = bo_matrix[end][start] = bond_order

def add_atoms_with_bonds(ace_mol,add_info_list):
    n = len(ace_mol.atom_list)
    new_bond_list = []
    new_atom_list = []
    cnt = 0
    for add_info in add_info_list:
        atom_type = add_info[0]
        bond_info = add_info[1]
        new_atom_list.append(atom_type)
        for bond in bond_info:
            start = n + cnt
            end = bond[0]
            bond_order = bond[1]
            new_bond_list.append((start,end,bond_order)) 
        cnt += 1
    add_atoms(ace_mol,new_atom_list)
    add_bonds(ace_mol,new_bond_list)

def molecule_to_intermediate(ace_mol):
    intermediate = chem.Intermediate()
    intermediate.atom_list = ace_mol.atom_list
    intermediate.bo_matrix = ace_mol.bo_matrix
    intermediate.adj_matrix = ace_mol.adj_matrix
    intermediate.atom_feature = ace_mol.atom_feature
    return intermediate

def molecule_to_ase_atoms(molecule):
    from ase import Atoms
    element_list = molecule.get_element_list()
    symbols = ''.join(element_list)
    positions = molecule.get_coordinate_list()
    #bo_matrix = molecule.get_matrix('bo')
    # also give original charge information
    #charge_list = molecule.get_chg_list().tolist()
    ase_atoms = Atoms(
            symbols,
            positions=positions, 
            charges=None,
            )
    #ase_atoms.set_initial_charges(charge_list)
    return ase_atoms

def read_geometries(directory):
    conformers = []
    with open(directory) as f:
        while True:
            try:
                atom_num = int(f.readline().strip())
                energy = float(f.readline().strip())
                atom_list = []
                for i in range(atom_num):
                    line = f.readline().strip().split()
                    element = line[0]
                    atom = chem.Atom(element)
                    atom.x = float(line[1])
                    atom.y = float(line[2])
                    atom.z = float(line[3])
                    atom_list.append(atom)
                molecule = chem.Molecule()
                molecule.atom_list = atom_list
                molecule.energy = energy
                conformers.append(molecule)
            except:
                break
    return conformers

def write_xyz_file_for_two_molecules(molecule_list,file_name,option='atomic number', scatter= True,scale=1.5):
    """
    Writes xyz files for two molecules
    
    :param 
    molecule_list: list of molecule class
    file_name: string. name of the file where you save
    option: 'atominc number', or 'element'.'atomic number' writes the element in atominc number and element writes the 'element' in element symbol
    scale: scale adjust the distance between two molecules 

    :return: None

    :log
    July 2nd-Jinwon Lee. writing
        
    """
    opnum = 0
    if (option=="atomic number"):
        pass
    elif (option=="element"):
        #print("you chose element option")
        opnum = 1
    else:
        print("Wrong writing option... Choose between 'atomic number' and 'element' ")

    total_atom_num = 0
    center_of_mass_list = []
    molecule_radius_list = []


    for molecule in molecule_list:
        total_atom_num += len(molecule.atom_list)
        center_of_mass_list.append(molecule.get_center_of_mass())
        molecule_radius_list.append( molecule.get_molecule_radius() )
    
    if scatter:
        tmp = np.array(center_of_mass_list[1]) - np.array(center_of_mass_list[0])    
        if (np.linalg.norm(tmp) < 0.2): # when two C.O.Ms are too close
            tmp = np.array([1,0,0])
        #print('tmp', tmp)
        norm = np.linalg.norm(tmp)
        scaler = sum(molecule_radius_list)*scale
        move_vector = tmp/norm*scaler 
        #print('mo', move_vector)
        move_molecule(molecule_list[1], move_vector)
    
    if file_name[-4:] != '.xyz':
        file_name += '.xyz'
    with open(file_name, "w") as f:
        f.write(str(total_atom_num)+ '\n' + '\n')
        for molecule in molecule_list:
            for atom in molecule.atom_list:
                if (opnum==0):
                    f.write(str(atom.atomic_number)+" "+str(atom.x)+" "+str(atom.y)+" "+str(atom.z)+'\n')
                else:
                    f.write(atom.get_element()+" "+str(atom.x)+" "+str(atom.y)+" "+str(atom.z)+'\n')

def scatter_molecules(molecule_list,grid = 0.3):
    n = len(molecule_list)
    position_list = dict()
    index = 0
    for i in range(n):
        molecule = molecule_list[i]
        radius = molecule.get_molecule_radius()
        center = molecule.get_center()
        if i>0:
            # Find position from the current position
            next_position = find_no_overlap(position_list,i-1,radius,grid)
            move_molecule(molecule,-center + next_position)
            position_list[i] = (next_position,radius)
        else:
            move_molecule(molecule,-center)
            position_list[0] = (np.array([0.0,0.0,0.0]),radius)
 
def check_atom_validity(group,bo,chg,octet=4):
    # lone pair inequality
    if group - bo - chg < 0:
        return False
    # Octet rule inequality
    if group + bo - chg > 2 * octet:
        return False
    return True


def get_molecule_group(adj_matrix,index=0):
    current_list = set([index])
    total_list = set([index])
    while len(current_list) > 0:
        new_current_list = set([])
        for i in current_list:
            neighbor_list = np.where(adj_matrix[i]>0)[0].tolist()
            new_current_list = new_current_list | set(neighbor_list)
        current_list = new_current_list - total_list
        total_list = total_list | new_current_list
    return total_list

def group_molecules(adj_matrix):
    n = len(adj_matrix)
    all_indices = set(range(n))
    groups = []
    index = 0
    while len(all_indices)>0:
        indices = get_molecule_group(adj_matrix,index)
        all_indices = all_indices - indices
        groups.append(list(indices))
        if len(all_indices)>0:
            index = min(all_indices)
        else:
            break
    return groups

def check_geometry(coordinate_list,criteria=0.5):
    distance_matrix = spatial.distance_matrix(coordinate_list,coordinate_list)
    np.fill_diagonal(distance_matrix,100)
    check_distance_matrix = np.where(distance_matrix < criteria,1,0)
    return np.sum(check_distance_matrix) < 1


def get_rmsd(molecule1,molecule2):
    n = len(molecule1.atom_list)
    if len(molecule2.atom_list) != n:
        print ('Cannot calculate RMSD!!!')
        return None
    coordinate_list1 = np.array(molecule1.get_coordinate_list())
    coordinate_list2 = np.array(molecule2.get_coordinate_list())
    rmsd = np.sqrt(np.sum((coordinate_list1 - coordinate_list2)**2)/n)
    return rmsd

def is_same_connectivity(original_molecule,new_molecule,max_coeff=1.3,min_coeff=0.95,space=0.01):
    coeff = min_coeff
    is_same = False
    if len(original_molecule.atom_list) == 1:
        return True, 0.95
    while coeff < max_coeff:
        adj_matrix = get_adj_matrix_from_distance(new_molecule,coeff)
        new_molecule.set_adj_matrix(adj_matrix)
        is_same = new_molecule.is_same_molecule(original_molecule,False)
        #print (adj_matrix - original_molecule.get_adj_matrix())
        if is_same:
            break
        coeff += space
    return is_same,coeff

def minimize_rmsd(reference_molecule,changing_molecule):
    from ase.build.rotate import minimize_rotation_and_translation
    reference_ase_atoms = molecule_to_ase_atoms(reference_molecule)
    target_ase_atoms = molecule_to_ase_atoms(changing_molecule)
    minimize_rotation_and_translation(reference_ase_atoms,target_ase_atoms)
    coordinate_list = target_ase_atoms.get_positions()
    locate_molecule(changing_molecule,coordinate_list)
    
def get_molecule_info_from_sdf(sdf_directory):
    from MetalloGen import globalvars as gv

    with open(sdf_directory,'r') as f:
        lines = f.readlines()

    # First, get the number of atoms
    n_atoms = int(lines[3].split()[0])
    n_bonds = int(lines[3].split()[1])

    z_list = []
    coords = []
    adj_matrix = np.zeros((n_atoms,n_atoms))
    chg_list = np.zeros(n_atoms)
    metal_index = None

    for i in range(n_atoms):
        line = lines[4+i].split()
        x = float(line[0])
        y = float(line[1])
        z = float(line[2])
        element = line[3]
        coords.append([x,y,z])
        z_list.append(chem.Atom(element).get_atomic_number())
        if element in gv.metal:
            metal_index = i
    for i in range(n_bonds):
        line = lines[4+n_atoms+i].split()
        s, e = line[0], line[1]
        s_ = int(s.strip())
        if s_ > n_atoms:
            tmp = s
            s = tmp[:3].strip()
            e = tmp[3:].strip()
        if not s.isdigit() or not e.isdigit():
            print("WRONG SDF; Error occurs during parsing bond block ...")
        s = int(s)-1
        e = int(e)-1
        adj_matrix[s][e] = 1
        adj_matrix[e][s] = 1
    chg_line = lines[4+n_atoms+n_bonds]
    if 'CHG' in chg_line:
        chg_line = chg_line.strip().split()
        n_chg = int(chg_line[2])
        for i in range(n_chg):
            idx = int(chg_line[3+2*i]) - 1
            chg = int(chg_line[4+2*i])
            chg_list[idx] = chg
    
    coords = np.array(coords)

    return z_list, coords, adj_matrix, chg_list, metal_index