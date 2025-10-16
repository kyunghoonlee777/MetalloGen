import numpy as np
import pickle as pkl

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

from MetalloGen import embed
from MetalloGen import chem, process
from MetalloGen.utils import ic
from MetalloGen.Calculator import orca, xtb_gaussian

def print_rd_geometry(rd_mol, positions):
    for i, rd_atom in enumerate(rd_mol.GetAtoms()):
        element = rd_atom.GetSymbol()
        x, y, z = positions[i]
        if abs(x) < 0.0001:
            x = 0.00
        if abs(y) < 0.0001:
            y = 0.00
        if abs(z) < 0.0001:
            z = 0.00
        print_x = f"{x:12.8f}"
        print_y = f"{y:12.8f}"
        print_z = f"{z:12.8f}"
        print(f"{element:<3} {print_x} {print_y} {print_z}")
    print()


class TMCOptimizer:
    
    def __init__(self, step_size = 0.1, calculator = None):

        if calculator is None:
            #calculator = orca.Orca()
            calculator = xtb_gaussian.XTB_Gaussian()
        self.step_size = step_size
        self.calculator = calculator  
        
        self.num_relaxation = 5
        self.maximal_displacement = 0.5
        self.adj_criteria = 0 # not used
        self.ratio_criteria = 0.6
        self.atom_d_criteria = 0.5
        self.success_criteria = 0.2 # not used
        self.bond_criteria = 1.2
        self.d_converge = 0.05 # Convergence criteria for FF opt
        self.fix_value = 20.0
        #self.fix_value = 50.0
        #self.binding_fix_value = 200.0
        self.binding_fix_value = 2000.0
        self.chk_file = 'scan.chk'
        self.scale_factor = {1:1, 2:1.1, 3:1.1, 4:1.1, 5:1.2, 6:1.2, 7:1.2, 8:1.2, 9:1.2, 'else':1.6}
        #self.ligand_d_criteria = 1.7


    def clean_geometry(self,metal_complex,scale=1.0,always_qc = False):
        print("Embedded geometry ...")
        metal_complex.print_coordinate_list()
        
        print("FF cleaning ...")
        try:
            ff_success = self.ff_clean(metal_complex,scale)
        except:
            print ('Internal failure for ff clean ...')
            ff_success = False

        if ff_success:
        #if True
            print("FF clean success!")
            print("FF cleaned geometry ...")
            metal_complex.print_coordinate_list()
            if always_qc:
                # Clean with QC ...
                qc_success = self.qc_clean(metal_complex,scale)
                return qc_success
            else:
                return True

        else:
            print("FF clean has failed ...")
            print("FF cleaned geometry ...")
            metal_complex.print_coordinate_list()
            print("Further cleaning with QC ...")
            #exit()
            # Clean with QC ...
            try:
                qc_success = self.qc_clean(metal_complex,scale)
            except:
                print ('Internal QC calculation fail ...')
                return False

            return qc_success


    def ff_clean(self,metal_complex,scale = 1.0):
        #ML_dist_dict = self.ML_dist_dict
        ligands = metal_complex.ligands
        atom_indices_for_each_ligand = metal_complex.get_atom_indices_for_each_ligand()
        center_atom = metal_complex.center_atom
        metal_r = center_atom.get_radius()
        metal_xyz = center_atom.get_coordinate()
       
        #print ('ligand',atom_indices_for_each_ligand)

        # Prepare FF setting ...    
        rd_mol_list = []
        tmp_positions = [] 
        scanning_indices = []
        target_values = dict()
        binding_groups_infos = dict()
        ligand_binding_group_infos = dict()
        cnt = 0
        max_value = 0.0        
        ligand_to_metal = dict()
        
        is_uff = False
        is_mmff = False
        
        step_size = self.step_size
        ratio_criteria = self.ratio_criteria
        atom_d_criteria = self.atom_d_criteria
        bond_criteria = self.bond_criteria
        d_converge = self.d_converge
        fix_value = self.fix_value
        binding_fix_value = self.binding_fix_value

        final_positions = metal_complex.get_position()
        ligand_atom_indices = []
        radius_list = [center_atom.get_radius()]
        ligand_adj_matrices = []


        # Gather ligand information for the scan ...
        for i in range(len(ligands)):       
            ligand = ligands[i]
            ace_mol = ligand.molecule.copy()
            for atom in ligand.molecule.atom_list:
                radius_list.append(atom.get_radius())

            try:
                valid_ace_mol = ace_mol.get_valid_molecule(False)
            except:
                valid_ace_mol = None

            if valid_ace_mol is None:
                valid_ace_mol = ace_mol.get_valid_molecule(False, 'xyz2mol')

            if len(valid_ace_mol.atom_list) == 1:
                valid_ace_mol.atom_list[0].set_element('Cl') # In case H- fail ...
                valid_ace_mol.atom_feature['chg'] = np.array([-1])
                valid_ace_mol.chg = -1
            bo_matrix = valid_ace_mol.get_bo_matrix()
            atom_list = ligand.molecule.atom_list
            if bo_matrix is None:
                valid_ace_mol.initialize()

            period_list, group_list = valid_ace_mol.get_period_group_list()

            # TODO: Change charge in each atom for octet deficient atoms (Remove if bad ...)
            n = len(valid_ace_mol.atom_list)
            for j in range(n):
                if period_list[j] == 1: # Pass for hydrogen
                    continue
                else: 
                    bo = np.sum(bo_matrix[j])
                    chg = valid_ace_mol.atom_feature['chg'][j]
                    g = group_list[j]
                    valence = g + bo - chg  # If valence less than 4 ...
                    if valence < 8:
                        valid_ace_mol.atom_feature['chg'][j] -= (8-valence)

            rd_mol = valid_ace_mol.get_rd_mol()
            rd_mol_list.append(rd_mol)
            atom_indices = atom_indices_for_each_ligand[i]        
            binding_infos = ligand.binding_infos # [[indices, geometric_idx]]

            tmp_indices = []
            for j in range(len(atom_indices)):
                tmp_positions.append(final_positions[atom_indices[j]])
                ligand_to_metal[len(tmp_positions)-1] = atom_indices[j]
                tmp_indices.append(cnt + j + 1) # Because metal goes to index 0
            ligand_atom_indices.append(tmp_indices)
            ligand_adj_matrices.append(valid_ace_mol.get_adj_matrix())
            total_binding_groups = []
            for info in binding_infos:
                sum_d = 0
                binding_groups = []
                for idx in info[0]:
                    binding_groups.append(cnt + idx)
                    coordinate = final_positions[atom_indices[idx]].tolist()
                    atom_r = atom_list[idx].get_radius()
                    sum_d += (metal_r + atom_r) * scale
                ref_d = sum_d / len(info[0]) 
                if len(info[0]) < 10:
                    ref_d *= self.scale_factor[len(info[0])] # Consider elongation of haptic interaction ...
                else:
                    ref_d *= 1.6
                target_values[tuple(binding_groups)] = ref_d
                binding_groups_infos[tuple(binding_groups)] = len(atom_indices)
                total_binding_groups.append(tuple(binding_groups))
                scanning_indices += binding_groups
            ligand_binding_group_infos[tuple(total_binding_groups)] = list(range(cnt, cnt+len(atom_indices)))
            cnt += len(atom_indices)

        # Construct original_ligand_adj_matrix ...
        original_ligand_adj_matrix = np.zeros((cnt+1,cnt+1)) 
        for k, atom_indices in enumerate(ligand_atom_indices):
            reduce_function = np.ix_(atom_indices, atom_indices)
            original_ligand_adj_matrix[reduce_function] = ligand_adj_matrices[k]

        num_scan = int(max_value/step_size) + 1

        combined_rd_mol = rd_mol_list[0]
        for rd_mol in rd_mol_list[1:]:
            combined_rd_mol = Chem.CombineMols(combined_rd_mol,rd_mol)

        Chem.SanitizeMol(combined_rd_mol)    
        AllChem.EmbedMolecule(combined_rd_mol, useRandomCoords = True, maxAttempts = 100, useBasicKnowledge = False, ignoreSmoothingFailures = True) 
        
        # Make force fields ...
        tmp_positions = np.array(tmp_positions)
        mmff = None
        uff = None
        try:
            mmff = AllChem.MMFFGetMoleculeForceField(combined_rd_mol, AllChem.MMFFGetMoleculeProperties(combined_rd_mol))
        except:
            pass
        try:
            uff = AllChem.UFFGetMoleculeForceField(combined_rd_mol)
        except:
            pass

        if mmff is None and uff is None:
            print("Force field is not supported ...")
            print(Chem.MolToSmiles(combined_rd_mol))
            return False

        n = len(radius_list)
        R = np.repeat(np.array(radius_list),n).reshape((n,n))
        R = R + R.T

        # Sort binding_groups by number of atoms ... (Small to large)
        ligand_binding_groups_list = list(ligand_binding_group_infos.keys())
        sorted_ligand_binding_groups_list = sorted(ligand_binding_group_infos, key=lambda x:len(ligand_binding_group_infos[x]))

        final_success = True
         
        # Scan by each ligand ...
        for ligand_idx,ligand_binding_groups in enumerate(sorted_ligand_binding_groups_list):
            for k in range(100):
                # Also, get old adj matrix ...
                old_positions = np.copy(tmp_positions)
                positions_with_metal = np.vstack((metal_xyz,old_positions))
                distance_matrix = cdist(positions_with_metal,positions_with_metal)
                np.fill_diagonal(distance_matrix,1e6)
                ratio_matrix = distance_matrix/R

                old_ligand_adj_matrix = np.where(ratio_matrix < bond_criteria, 1, 0)
                old_ligand_adj_matrix[0,:] = 0
                old_ligand_adj_matrix[:,0] = 0

                # Translation ...
                abs_delta = 0
                current_binding_indices = []
                for binding_groups in list(ligand_binding_groups):
                    binding_vectors = []
                    for idx in binding_groups:
                        binding_vectors.append(tmp_positions[idx].tolist())
                    binding_vectors = np.array(binding_vectors)
                    ref_d = target_values[tuple(binding_groups)]
                    v = np.mean(binding_vectors,axis=0)
                    d = np.linalg.norm(v)
                    delta_d = d - ref_d
                    current_binding_indices += list(binding_groups)

                    # Adjust to ref_d 
                    if delta_d > step_size:
                        delta_d = step_size
                    elif delta_d < -step_size:
                        delta_d = -step_size
                    if abs_delta < abs(delta_d):
                        abs_delta = delta_d
                    for idx in binding_groups:
                        tmp_positions[idx] -= delta_d * v/d

                if k > 0 and abs_delta < d_converge: # Must perform at least one FF opt
                    break

                ff_success = False
                if len(scanning_indices) < len(tmp_positions):
                    conformer = combined_rd_mol.GetConformer()
                    for i, position in enumerate(tmp_positions):
                        x, y, z = position
                        conformer.SetAtomPosition(i, Point3D(x,y,z))    

                    # Set force fields ... 
                    mmff = AllChem.MMFFGetMoleculeForceField(combined_rd_mol, AllChem.MMFFGetMoleculeProperties(combined_rd_mol))
                    uff = AllChem.UFFGetMoleculeForceField(combined_rd_mol)
                    
                    # FF opt    
                    if mmff is not None:
                        mmff.Initialize()
                        #for atom_idx in scanning_indices:
                            #mmff.AddFixedPoint(atom_idx) # Fix binding atoms ...
                        for atom_idx in range(len(tmp_positions)):
                            force_constant = fix_value
                            if atom_idx in scanning_indices:
                                force_constant = binding_fix_value
                            mmff.MMFFAddPositionConstraint(atom_idx,maxDispl=0.00,forceConstant = force_constant)

                    if uff is not None:
                        uff.Initialize()
                        #for atom_idx in scanning_indices:
                        #    uff.AddFixedPoint(atom_idx) # Fix binding atoms ...
                        
                        for atom_idx in range(len(tmp_positions)):
                            force_constant = fix_value
                            if atom_idx in scanning_indices:
                                force_constant = binding_fix_value
                            uff.UFFAddPositionConstraint(atom_idx,maxDispl=0.00,forceConstant = force_constant)

                    ffs = [mmff,uff] 
                    for ff in ffs:
                        if ff is None:
                            continue
                        try:
                            ff.Minimize()
                        except:
                            continue
                        
                        conformer = combined_rd_mol.GetConformer()
                        tmp_positions = conformer.GetPositions()
                 
                        # Check validity of the geometry (tmp_positions)
                        # Insert metal at zero ...
                        positions_with_metal = np.vstack((metal_xyz,tmp_positions))
                        distance_matrix = cdist(positions_with_metal,positions_with_metal)
                        np.fill_diagonal(distance_matrix,1e6)
                        ratio_matrix = distance_matrix/R
                        min_ratio = np.min(ratio_matrix)
                        
                        # Check the Collapse of geometry ...
                        if min_ratio < ratio_criteria or not np.all(distance_matrix) > atom_d_criteria: 
                            print("[FF Scan] Atoms are too close ... Restoring to the original positions !")
                            print_rd_geometry(combined_rd_mol,tmp_positions)
                            tmp_positions = old_positions # Restore to original ...
                            continue
                        
                        # Check the ratio between metal and the binding indices ...
                        tmp_indices = [i+1 for i in current_binding_indices]
                        min_ratio = np.min(ratio_matrix[0,tmp_indices])
                        min_distance = np.min(distance_matrix[0,tmp_indices])
                        ligand_indices = ligand_binding_group_infos[ligand_binding_groups]
                        tmp_indices = [i+1 for i in ligand_indices]
                        total_min_ratio = np.min(ratio_matrix[0,tmp_indices])
                        total_min_distance = np.min(distance_matrix[0,tmp_indices])
                        if min_ratio < bond_criteria:
                            if min_ratio > total_min_ratio + 0.1 or min_distance > total_min_distance + 0.2: # May change ...
                                print("[FF scan] Other atoms are likely to bind to the metal ... Using the previous positions !")
                                print_rd_geometry(combined_rd_mol,tmp_positions)
                                tmp_positions = old_positions # Restore to original ...
                                continue
                        
                        # Check the distance between ligands ...
                        ligand_adj_matrix = np.where(ratio_matrix < bond_criteria, 1, 0)
                        ligand_adj_matrix[0,:] = 0
                        ligand_adj_matrix[:,0] = 0
                        delta_matrix = ligand_adj_matrix - old_ligand_adj_matrix
                        formed_bonds = np.stack(np.where(delta_matrix > 0),axis=1).tolist()
                        removed_bonds = np.stack(np.where(delta_matrix < 0),axis=1).tolist()
                        # Compare with the original ligand adj matrix
                        adj_change = False
                        for bond in formed_bonds:
                            s, e  = bond
                            if original_ligand_adj_matrix[s][e] == 0:
                                adj_change = True
                                break

                        if not adj_change:
                            for bond in removed_bonds:
                                s, e = bond
                                if original_ligand_adj_matrix[s][e] > 0:
                                    adj_change = True
                                    break

                        if adj_change:
                            print('[FF Scan] Adjacent matrix has changed ... Restoring to the original positions !')
                            print_rd_geometry(combined_rd_mol,tmp_positions)
                            tmp_positions = old_positions
                            for bond in formed_bonds + removed_bonds:
                                s, e = bond
                                if s < e:
                                    continue
                                    #print (s,e,distance_matrix[s][e],ratio_matrix[s][e])
                            continue
                            
                        else:
                            ff_success = True
                            break
                else:
                    ff_success = True

                if not ff_success:
                    final_success = False
                    break
                
        # check move_dict to determine the success of FF clean 
        # Less than int(success_criteria/step_size)+1 should be left for successful clean ...

        # If fine, update final positions
        for i in ligand_to_metal:
            x,y,z = tmp_positions[i]
            final_positions[ligand_to_metal[i]] = [x,y,z]
        # Update ligand ...
        metal_complex.set_position(final_positions)
        return final_success

    def qc_clean(self,metal_complex,scale = 1.0):
        ligands = metal_complex.ligands
        atom_indices_for_each_ligand = metal_complex.get_atom_indices_for_each_ligand()
        center_atom = metal_complex.center_atom
        metal_r = center_atom.get_radius()
        metal_index = metal_complex.metal_index
        total_atom_list = metal_complex.get_atom_list()
        chg = metal_complex.chg
        mult = metal_complex.multiplicity

        step_size = self.step_size
        calculator = self.calculator
        num_relaxation = self.num_relaxation
        maximal_displacement = self.maximal_displacement
        ratio_criteria = self.ratio_criteria
        atom_d_criteria = self.atom_d_criteria
        bond_criteria = self.bond_criteria
        
        radius_list = [atom.get_radius() for atom in total_atom_list]
        n = len(radius_list)
        R = np.repeat(np.array(radius_list),n).reshape((n,n))
        R = R + R.T
        
        # Make constraints
        fixing_atoms = [metal_index]
        max_value = 0
        translation_vectors = dict()
        
        # Initialize ligand ...
        n = len(ligands)
        for i in range(n):
            ligand = ligands[i]
            atom_indices = atom_indices_for_each_ligand[i]
            
            molecule = ligand.molecule
            atom_list = molecule.atom_list
            
            binding_infos = ligand.binding_infos
            # Update constraint for binding atoms ...
            for binding_info in binding_infos:
                binding_vectors = []
                binding_groups = []
                sum_d = 0
                for index in binding_info[0]:
                    binding_groups.append(atom_indices[index])
                    x, y, z = atom_list[index].get_coordinate().tolist()
                    binding_vectors.append([x, y, z])
                    atom_r = atom_list[index].get_radius()
                    metal_r = center_atom.get_radius()
                    sum_d += (metal_r + atom_r) * scale
                ref_d = sum_d / len(binding_info[0]) 
                if len(binding_info[0]) < 10:
                    ref_d *= self.scale_factor[len(binding_info[0])]
                else:
                    ref_d *= 1.6

                fixing_atoms += binding_groups
                binding_vectors = np.array(binding_vectors)
                v = np.mean(binding_vectors,axis=0)
                d = np.linalg.norm(v)
                delta_d = d - ref_d
                # step size to 0.1
                if max_value < abs(delta_d): # Used for determinining max step
                    max_value = abs(delta_d)
                translation_vectors[tuple(binding_groups)] = (v/d,delta_d)
        
        # Make intermediate and scan
        num_scan = int(max_value/step_size) + 1
        tmp_molecule = metal_complex.get_molecule()
        fixing_atoms = list(set(fixing_atoms))
        fixing_atoms = [[i] for i in fixing_atoms]

        # Start scan ...
        try:
            energy_list = [calculator.get_energy(tmp_molecule)]
        except:
            print("Energy calculation failed !!!")
            return False

        # Get original_ligand_adj_matrix ...
        original_ligand_adj_matrix = np.copy(metal_complex.get_adj_matrix())
        original_ligand_adj_matrix[metal_index,:] = 0
        original_ligand_adj_matrix[:,metal_index] = 0

        for i in range(num_scan):

            # Get current ligand adj matrix ...
            prev_positions = tmp_molecule.get_coordinate_list()
            tmp_positions = np.copy(prev_positions)
            distance_matrix = cdist(tmp_positions, tmp_positions)
            ratio_matrix = distance_matrix/R
            old_ligand_adj_matrix = np.where(ratio_matrix < bond_criteria - 0.15,1,0)
            old_ligand_adj_matrix[metal_index,:] = 0
            old_ligand_adj_matrix[:,metal_index] = 0

            # Pull ligands ...
            for binding_group in translation_vectors:
                v, delta_d = translation_vectors[binding_group]
                for j in list(binding_group):
                    tmp_positions[j] -= v * delta_d/num_scan
            process.locate_molecule(tmp_molecule,tmp_positions)

            calculator.relax_geometry(tmp_molecule,fixing_atoms,num_relaxation=num_relaxation,file_name=f'qc_relax_{i+1}',maximal_displacement=maximal_displacement)
            energy = tmp_molecule.energy

            if energy is None:
                print("Calculation was not successful ...")
                return False
            new_positions = tmp_molecule.get_coordinate_list()
            
            # Check the validity of the geometry
            distance_matrix = cdist(new_positions,new_positions)
            np.fill_diagonal(distance_matrix,1e6)
            ratio_matrix = distance_matrix/R
            min_ratio = np.min(ratio_matrix)
            m = len(ratio_matrix)

            # Check collapse between atoms ...
            if min_ratio < ratio_criteria or not np.all(distance_matrix) > atom_d_criteria: 
                print("[QC Scan] Atoms are too close ... Using the previous positions !")
                tmp_molecule.print_coordinate_list()
                process.locate_molecule(tmp_molecule,prev_positions)
                tmp_molecule.energy = energy_list[-1]
                break

            # Check the distance between ligands ...
            ligand_adj_matrix = np.where(ratio_matrix < bond_criteria - 0.15, 1, 0)
            ligand_adj_matrix[metal_index,:] = 0
            ligand_adj_matrix[:,metal_index] = 0
            delta_matrix = ligand_adj_matrix - old_ligand_adj_matrix
            formed_bonds = np.stack(np.where(delta_matrix > 0),axis=1)
            removed_bonds = np.stack(np.where(delta_matrix < 0),axis=1)
            # Compare with the original ligand adj matrix
            adj_change = False
            for bond in formed_bonds:
                s, e  = bond
                if original_ligand_adj_matrix[s][e] == 0:
                    adj_change = True
                    break

            if not adj_change:
                for bond in removed_bonds:
                    s, e = bond
                    if original_ligand_adj_matrix[s][e] > 0:
                        adj_change = False
                        break

            # If change in adj matrix is detected ...
            if adj_change:
                print (formed_bonds, removed_bonds)
                print('[QC Scan] Adjacent matrix changed ... Stop Scanning !!!')
                tmp_molecule.print_coordinate_list()
                process.locate_molecule(tmp_molecule,prev_positions) # Restore to old positions ...
                tmp_molecule.energy = energy_list[-1]
                break
            else:
                energy_list.append(energy)
            print("Scan:",i)
            print(energy)
            tmp_molecule.print_coordinate_list()

        print("Scan finished !!!")

        # Final translation
        translate_vector = - center_atom.get_coordinate()
        process.translate_molecule(tmp_molecule,translate_vector)
        
        coordinate_list = tmp_molecule.get_coordinate_list()
        metal_complex.set_position(coordinate_list)

        # Check whether scan was done near to the destination ...
        return i > num_scan - 4
