### python provided modules ###
import time
import os
import sys
from copy import deepcopy
import subprocess
import pickle
import datetime
import argparse
import numpy as np
import distutils.spawn


### Module for reading gaussian files ###
import cclib

### ace-reaction libraries ###
import chem
import process

def get_coords_from_out(file_directory):
    # count number of instances of 
    # "CARTESIAN COORDINATES"
    count = 0
    with open(file_directory, 'r') as f:
        for line in f.readlines():
            if line.strip() == "CARTESIAN COORDINATES":
                count += 1
    new_count = 0
    with open(file_directory, 'r') as f:
        for line_no, line in enumerate(f.readlines()):
            if line.strip() == "CARTESIAN COORDINATES":
                new_count += 1
                if new_count == count:
                    break
    coords = list()
    with open(file_directory, 'r') as f:
    # read from this point on
        for line in f.readlines()[line_no+2:]:
            if line.strip() == '':
                break
            line = line.split()
            x = float(line[2])
            y = float(line[3])
            z = float(line[4])
            coords.append((x, y, z))

    return coords

def angle(p1, p2, p3):
    """Returns angle in degrees"""
    # p2 -> p1 -> p3
    v1 = p1 - p2
    v2 = p3 - p2
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    radians = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return radians * 180.0 / np.pi

def dihedral_angle(p1, p2, p3, p4):
    """Returns angle in degrees"""
    # p1 -> p2 -> p3 -> p4, with p2 -> p3 axis
    v1 = p1 - p2
    v2 = p2 - p3
    v3 = p3 - p4
    u1 = np.cross(v1, v2)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v2, v3)
    u2 /= np.linalg.norm(u2)
    sign = np.sign(np.dot(u2, v2))
    radians = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
    if sign != 0:
        radians *= sign
    return radians * 180.0 / np.pi

# p1 = np.array([-0.565165, -0.282022, 0.441409])
# p2 = np.array([0.417187, 0.564837, 0.214351])
# p3 = np.array([1.359538, -0.162184, -0.363409])
# p4 = np.array([0.764, 0.069, -2.88])
#
# print(dihedral_angle(p1, p2, p3, p4))
#
# import os
# os.system("mopac test.mop")
# quit()


class Mopac:

    def __init__(self,command='mopac',working_directory=None):
        # Check command
        check = distutils.spawn.find_executable(command)
        if check is not None:
            self.command = command
        else:
            # Check default command
            commands = ['mopac']
            found = False
            for old_command in commands:
                check = distutils.spawn.find_executable(old_command)
                if check is not None:
                    self.command = old_command
                    print (f'command {self.command} is used for running MOPAC, instead of {command}!')
                    found = True
                    break
            if not found:
                print ('MOPAC not found!')
                exit()

        if working_directory is not None:
            if not os.path.exists(working_directory):
                os.system(f'mkdir {working_directory}')
                if not os.path.exists(working_directory):
                    #print ('Defined working directory does not exist!!!')
                    working_directory = None

        if working_directory is None:
            working_directory = os.getcwd()
            #print ('working directory is set to current diretory:',working_directory)

        self.working_directory = working_directory
        # Check save directory        
        self.content='pm6 '
        self.energy_unit = 'Hartree'
        self.basis = ''

    def __str__(self):
        content = True
        content = content + f'working_directory: {self.working_directory}\n'
        content = content + f'command: {self.command}\n'
        content = content + f'Energy: {self.energy_unit}\n\n'
        content = content + f'###### qc_input ######\n{self.content}\n'
        return content

    def load_content(self,template_directory):
        if os.path.exists(template_directory):
            content = ''
            with open(template_directory) as f:        
                lines = f.readlines()
                n = len(lines)
                for i,line in enumerate(lines):
                    if i == n-1:
                        line = line.strip()
                    content = content + line
            self.content = content
        else:
            print ('qc input file does not exist!!!')

    # def load_basis(self,basis_directory):
    #     if os.path.exists(basis_directory):
    #         basis = ''
    #         with open(basis_directory) as f:        
    #             for line in f:
    #                 basis = basis + line
    #         self.basis = basis
    #     else:
    #         print ('basis file does not exist!!!')

    def change_working_directory(self,working_directory):
        # Get current reaction coordinate
        if not os.path.exists(working_directory):
            print ('Working directory does not exist! Creating new directory ...')
            os.system(f'mkdir {working_directory}')
            if not os.path.exists(working_directory):
                print ('Cannot create working directory!!!\n Recheck your working directory ...')
                exit()
            else:
                print ('working directory changed into:',working_directory)
                self.working_directory = working_directory
        else:
            print ('working directory changed into:',working_directory)
            self.working_directory = working_directory

    def get_content(self):
        return self.content 

    def get_default_mol_params(self,molecule):
        try:
            chg = molecule.get_chg()
        except:
            chg = 0
        try:
            multiplicity = molecule.get_multiplicity()
        except:
            multiplicity = None
        if multiplicity is None:
            try:
                e_list = molecule.get_num_of_lone_pair_list()
                num_of_unpaired_e = len(np.where((2*e_list) % 2 == 1)[0])    
                multiplicity = num_of_unpaired_e + 1
            except:
                z_sum = np.sum(molecule.get_z_list())
                multiplicity = (z_sum - chg) % 2 + 1
        return chg,multiplicity

    def write_molecule_info(self,f,molecule, fixed_coords):
        for atom in molecule.atom_list:
            if fixed_coords:
                atom_string = f'{atom.get_element()} {atom.x:.6f} 0 {atom.y:.6f} 0 {atom.z:.6f} 0\n'
            else:
                atom_string = f'{atom.get_element()} {atom.x:.6f} 1 {atom.y:.6f} 1 {atom.z:.6f} 1\n'
            f.write(atom_string)
        f.write('\n')

    def make_input(self,
                   molecules,
                   fixed_coords,
                   chg=None,
                   multiplicity=None,
                   file_name='test',
                   constraints={},
                   extra=''):
        f = open(f'{file_name}.mop','w')
        if chg is None or multiplicity is None:
            if molecules[0] is not None:
                chg, multiplicity = self.get_default_mol_params(molecules[0])
            elif molecules[-1] is not None:
                chg, multiplicity = self.get_default_mol_params(molecules[-1])
        content = self.get_content()
        # MOPAC uses spin_state instead of spin multiplicity
        # Ex: MS=0.5 is doublet
        quo, rem = divmod(multiplicity, 2)
        if rem == 0:
            spin_state = f"{quo - 0.5:.1f}"
        else:
            spin_state = f"{quo}"
        content = content + f' charge={chg} MS={spin_state} ' + f'{extra}\n'
        f.write(content)
        labels = ['R','P','TS']
        for i, molecule in enumerate(molecules):
            if molecule is not None:
                f.write(f'{labels[i]}\ntest\n')
                self.write_molecule_info(f,molecule,fixed_coords)
        f.close()

    def move_file(self,file_name,save_directory):
        if file_name is not None and save_directory is not None:
            # file_directory = f'{file_name}.mop'
            os.system(f'mv {file_name}.mop {save_directory}/')
            os.system(f'mv {file_name}.arc {save_directory}/')
            os.system(f'mv {file_name}.out {save_directory}/')

    def run(self,molecule,chg=None,multiplicity=None,file_name='test',save_directory=None): # Running with user defined result and return raw data
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        self.make_input([molecule],False,chg,multiplicity,file_name,{},'')
        os.system(f'{self.command} {file_name}.mop > {file_name}.log')
        # Read output
        p = cclib.parser.MOPAC(f'{file_name}.out')
        data = p.parse()
        self.move_file(file_name,save_directory)
        converter = 1
        #try:
        #    normal_termination = data.metadata['success']
        #except:
        #    normal_termination = False
        #print (normal_termination)
        if self.energy_unit == 'kcal':
            converter = 23.06
        if self.energy_unit == 'Hartree':
            converter = 0.036749326681
        if 'ef' in self.content: 
            #print ('Running optimization !!!')
            #print (data.scfenergies)
            try:
                data.scfenergies
            except:
                print ('Calculation not finished !!!', molecule.energy)
                return False
            index = np.argmin(data.scfenergies)
            if index > len(data.atomcoords) - 1:
                index = -1
            coordinate_list = data.atomcoords[index] # Sometimes length is not matched, it performs extra scfenergy calculation
            energy = data.scfenergies[index] * converter
            process.locate_molecule(molecule,coordinate_list,False)
            molecule.energy = energy
            #print('Optimized energy:',molecule.energy)
        os.chdir(current_directory) 
        return True

    def get_energy(self,molecule,chg=None,multiplicity=None,file_name='sp',extra='',save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        if 'noopt' not in extra:
            extra = ' noopt ' + extra
        if 'geo-ok' not in extra:
            extra = ' geo-ok ' + extra
        self.make_input([molecule],True,chg,multiplicity,file_name=file_name,extra=extra)
        os.system(f'{self.command} {file_name}.mop > {file_name}.log')

        # Read output
        p = cclib.parser.MOPAC(f'{file_name}.out')
        data = p.parse()
        self.move_file(file_name,save_directory)
        converter = 1
        os.chdir(current_directory)
        if self.energy_unit == 'kcal':
            converter = 23.06
        if self.energy_unit == 'Hartree':
            converter = 0.036749326681
        #os.system('mv new.chk old.chk')
        os.chdir(current_directory)
        return converter*data.scfenergies[-1]

    def optimize_geometry(self, molecule, constraints={},
                          chg=None, multiplicity=None, file_name='test',
                          extra='', save_directory=None):
        # constrained opt is handled separately
        if len(constraints) != 0:
            return self.optimize_geometry_with_contraints(molecule,
                                                          constraints=constraints,
                                                          chg=chg,
                                                          multiplicity=multiplicity,
                                                          file_name=file_name,
                                                          extra=extra,
                                                          save_directory=save_directory)

        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        self.make_input([molecule],False,chg,multiplicity,file_name,extra)
        os.system(f'{self.command} {file_name}.mop > {file_name}.log')

        # Read output
        p = cclib.parser.MOPAC(f'{file_name}.out')
        data = p.parse()
        self.move_file(file_name,save_directory)

        # Get minimal energy geometry
        index = np.argmin(data.scfenergies)
        if index > len(data.atomcoords) - 1:
            index = -1
        coordinate_list = data.atomcoords[index] # Sometimes length is not matched, it performs extra scfenergy calculation
        converter = 1
        if self.energy_unit == 'kcal':
            converter = 23.06
        if self.energy_unit == 'Hartree':
            converter = 0.036749326681
            #converter = 1/27.2114
        energy = data.scfenergies[index] * converter
        process.locate_molecule(molecule,coordinate_list,False)
        molecule.energy = energy
        os.chdir(current_directory)

    def optimize_geometry_with_contraints(self, molecule, constraints={}, 
                                          chg=None, multiplicity=None,
                                          file_name='test', extra='', save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        # with constraints, constrained atoms must be below unconstrained atoms
        # in the input file
        mol_atom_list = molecule.get_atom_list()
        num_atoms = len(mol_atom_list)
        constrained_indices = dict()
        for idx in range(num_atoms):
            for constraint in constraints:
                if not hasattr(constraint, '__iter__'):
                    constraint_list = list([constraint])
                else:
                    constraint_list = list(constraint)
                if idx in constraint_list:
                    if idx not in constrained_indices:
                        constrained_indices[idx] = len(constraint_list)
                    else:
                        if constrained_indices[idx] > len(constraint_list):
                            constrained_indices[idx] = len(constraint_list)
        pos_constrained_indices = [idx for idx in constrained_indices
                                   if constrained_indices[idx] == 1]
        len_constrained_indices = [idx for idx in constrained_indices
                                   if constrained_indices[idx] == 2]
        ang_constrained_indices = [idx for idx in constrained_indices
                                   if constrained_indices[idx] == 3]
        dih_constrained_indices = [idx for idx in constrained_indices
                                   if constrained_indices[idx] == 4]
        assert len(ang_constrained_indices) == 0
        assert len(dih_constrained_indices) == 0

        unconstrained_indices = [idx for idx in range(num_atoms)
                                 if idx not in constrained_indices]
        
        

        # reordering is performed below
        reordered_input = list()
        map_reordered_to_original = dict()
        map_original_to_reordered = dict()
        reordered_idx = 0
        exhausted_indices = list()

        # add three dummy atoms
        max_coord = 0
        for idx in range(num_atoms):
            max_val = max(mol_atom_list[idx].get_coordinate())
            if max_coord < max_val:
                max_coord = max_val
        max_coord += 1.0
        # dummy at (max_coord, 0, 0) and (0, max_coord, 0)
        values = ['XX', max_coord, 0, 0.0, 0, 0.0, 0]
        reordered_input.append({'original_idx': None,
                                'type': 'cartesian',
                                'values': values})
        values = ['XX', 0.0, 0, max_coord, 0, 0.0, 0]
        reordered_input.append({'original_idx': None,
                                'type': 'cartesian',
                                'values': values})
        values = ['XX', 0.0, 0, 0.0, 0, max_coord, 0]
        reordered_input.append({'original_idx': None,
                                'type': 'cartesian',
                                'values': values})

        for original_idx in pos_constrained_indices:
            exhausted_indices.append(original_idx)
            map_reordered_to_original[reordered_idx] = original_idx
            map_original_to_reordered[original_idx] = reordered_idx
            element = mol_atom_list[original_idx].get_element()
            # coords is a size 3 numpy array
            coords = mol_atom_list[original_idx].get_coordinate()
            values = [element, coords[0], 0, coords[1], 0, coords[2], 0]
            reordered_input.append({'original_idx': original_idx,
                                    'type': 'cartesian',
                                    'values': values})
            reordered_idx += 1
        for original_idx in unconstrained_indices:
            exhausted_indices.append(original_idx)
            map_reordered_to_original[reordered_idx] = original_idx
            map_original_to_reordered[original_idx] = reordered_idx
            element = mol_atom_list[original_idx].get_element()
            # coords is a size 3 numpy array
            coords = mol_atom_list[original_idx].get_coordinate()
            values = [element, coords[0], 1, coords[1], 1, coords[2], 1]
            reordered_input.append({'original_idx': original_idx,
                                    'type': 'cartesian',
                                    'values': values})
            reordered_idx += 1

        len_constraints = {key: value for key, value in deepcopy(constraints).items()
                           if hasattr(key, '__iter__') and len(key) == 2}

        # at least one atom per loop must be defined
        # separately to avoid infinite loop
        loops = set()
        for atom_idx in len_constrained_indices:
            next_atom = False
            for loop in loops:
                if atom_idx in loop:
                    next_atom = True
                    break
            if next_atom:
                continue
            new_loop = set()
            new_loop.add(atom_idx)
            current_idx = atom_idx
            while True:
                break_loop = False
                for constraint in len_constraints:
                    if current_idx not in constraint:
                        continue
                    other_idx = list(deepcopy(constraint))
                    other_idx.remove(current_idx)
                    other_idx = other_idx[0]
                    if other_idx != atom_idx:
                        new_loop.add(other_idx)
                        current_idx = other_idx
                    else:
                        # loop found
                        break_loop = True
                        break
                if break_loop:
                    ordered_loop = sorted(list(new_loop))
                    loops.add(tuple(ordered_loop))
                    break

        # combine loops with same atom
        combined_loops = set()
        for atom_idx in len_constrained_indices:
            combined_loop = list()
            for loop in loops:
                if atom_idx not in loop:
                    continue
                combined_loop += list(loop)
            combined_loop = list(set(combined_loop))
            combined_loop = tuple(sorted(combined_loop))
            combined_loops.add(combined_loop)

        for loop in combined_loops:
            # first atom of a loop defined with cartesian coords
            original_idx = loop[0]
            map_reordered_to_original[reordered_idx] = original_idx
            map_original_to_reordered[original_idx] = reordered_idx
            element = mol_atom_list[original_idx].get_element()
            # coords is a size 3 numpy array
            coords = mol_atom_list[original_idx].get_coordinate()
            values = [element, coords[0], 1, coords[1], 1, coords[2], 1]
            reordered_input.append({'original_idx': original_idx,
                                    'type': 'cartesian',
                                    'values': values})
            exhausted_indices.append(original_idx)
            len_constrained_indices.remove(original_idx)
            reordered_idx += 1

        while len(len_constrained_indices) > 0:
            original_idx = len_constrained_indices[0]
            for constraint in len_constraints:
                if original_idx not in constraint:
                    continue
                if original_idx in exhausted_indices:
                    continue

                constrained_len = len_constraints[constraint]
                exhausted_idx = list(deepcopy(constraint))
                exhausted_idx.remove(original_idx)
                exhausted_idx = exhausted_idx[0]

                if exhausted_idx not in exhausted_indices:
                    continue

                # atom 1 (to be added in reordered_input)
                atom_1 = mol_atom_list[original_idx]
                atom_1_pos = atom_1.get_coordinate()
                # atom 2 (already included in reordered_input)
                atom_2 = mol_atom_list[exhausted_idx]
                atom_2_pos = atom_2.get_coordinate()
                # atom 3 and atom 4: dummy atoms
                # atom 3 and 4
                atom_3_pos = np.array([max_coord, 0.0, 0.0])
                atom_4_pos = np.array([0.0, max_coord, 0.0])
                # get distance between 1 and 2
                # dist = np.linalg.norm(atom_1_pos-atom_2_pos)
                # get angle between 1, 2, and 3
                ang = angle(atom_1_pos,
                            atom_2_pos, 
                            atom_3_pos)
                # get dihedral angle between 1, 2, 3, and 4
                dih = dihedral_angle(atom_1_pos,
                                     atom_2_pos,
                                     atom_3_pos,
                                     atom_4_pos)

                reordered_exhausted_idx = map_original_to_reordered[exhausted_idx]

                # write coordinate
                # optimization flags: 0 = fixed, 1 = not fixed
                element = mol_atom_list[original_idx].get_element()
                values = [element, constrained_len, 0, ang, 1, dih, 1, 
                          reordered_exhausted_idx+1+3, 1, 2]
                reordered_input.append({'original_idx': original_idx,
                                        'type': 'internal',
                                        'values': values})
                map_reordered_to_original[reordered_idx] = original_idx
                map_original_to_reordered[original_idx] = reordered_idx
                exhausted_indices.append(original_idx)
                len_constrained_indices.remove(original_idx)
                reordered_idx += 1


        for info in reordered_input:
            print(info)
        assert num_atoms + 3 == len(reordered_input)

        # make input here
        f = open(f'{file_name}.mop','w')
        if chg is None or multiplicity is None:
            if molecule is not None:
                chg, multiplicity = self.get_default_mol_params(molecule)
        content = self.get_content()
        # MOPAC uses spin_state instead of spin multiplicity
        # Ex: MS=0.5 is doublet
        quo, rem = divmod(multiplicity, 2)
        if rem == 0:
            spin_state = f"{quo - 0.5:.1f}"
        else:
            spin_state = f"{quo}"
        content = content + f' charge={chg} MS={spin_state} ' + f'{extra}\n'
        f.write(content)
        f.write(f'Constrained Opt\ntest\n')
        for coords_dict in reordered_input:
            if coords_dict['type'] == 'cartesian':
                coords = coords_dict['values']
                f.write(f"{coords[0]:<3} ")
                f.write(f"{coords[1]:15.8f} ")
                f.write(f"{coords[2]} ")
                f.write(f"{coords[3]:15.8f} ")
                f.write(f"{coords[4]} ")
                f.write(f"{coords[5]:15.8f} ")
                f.write(f"{coords[6]}\n")
            elif coords_dict['type'] == 'internal':
                coords = coords_dict['values']
                f.write(f"{coords[0]:<3} ")
                f.write(f"{coords[1]:15.8f} ")
                f.write(f"{coords[2]} ")
                f.write(f"{coords[3]:15.8f} ")
                f.write(f"{coords[4]} ")
                f.write(f"{coords[5]:15.8f} ")
                f.write(f"{coords[6]} ")
                f.write(f"{coords[7]} ")
                f.write(f"{coords[8]} ")
                f.write(f"{coords[9]}\n")
        f.close()

        os.system(f'{self.command} {file_name}.mop > {file_name}.log')

        # Read output
        coords = get_coords_from_out(f"{file_name}.out")
        self.move_file(file_name,save_directory)
        p = cclib.parser.MOPAC(f'{file_name}.out')
        data = p.parse()
        try:
            index = np.argmin(data.scfenergies)

            if self.energy_unit == 'kcal':
                converter = 23.06
            if self.energy_unit == 'Hartree':
                converter = 0.036749326681
                #converter = 1/27.2114
            energy = data.scfenergies[index] * converter
            molecule.energy = energy
        except:
            index = 0


        # reorder_coords
        reordered_coords = [None for _ in range(num_atoms)]
        for key, val in map_reordered_to_original.items():
            reordered_coords[val] = coords[key]

        process.locate_molecule(molecule,reordered_coords,False)
        os.chdir(current_directory)
        
    # def relax_geometry(self,molecule,constraints,chg=None,multiplicity=None,file_name='test',num_relaxation=5,maximal_displacement=1000,save_directory=None):
    #     if maximal_displacement < 100:
    #         max_step = int(maximal_displacement*100) + 1
    #         if len(constraints) > 0:
    #             extra = f' opt(modredundant,loose,maxcycles={num_relaxation},maxstep={max_step},notrust) Symmetry=None'
    #         else:
    #             extra = f' opt(loose,maxcycles={num_relaxation},maxstep={max_step},notrust) Symmetry=None'
    #     else:
    #         if len(constraints) > 0:
    #             extra = f' opt(modredundant,loose,maxcycles=15) Symmetry = None'
    #         else:
    #             extra = f' opt(loose,maxcycles=15) Symmetry = None'
    #     self.optimize_geometry(molecule,constraints,chg,multiplicity,file_name,extra,save_directory)


    # def search_ts(self,molecules,chg=None,multiplicity=None,method='qst2',file_name=None,extra = '',save_directory=None,check_frequency = True):
    #     current_directory = os.getcwd()
    #     os.chdir(self.working_directory)
    #     n = 0
    #     for molecule in molecules:
    #         if molecule is not None:
    #             n += 1
    #     if n == 1:
    #         ts_molecule = molecules[-1].copy()
    #     else:
    #         ts_molecule = molecules[0].copy()
    #     if 'opt' not in extra:
    #         if 'qst' in method:
    #             extra = f' opt(qst{n},maxcycles=1000) freq ' + extra + ' '
    #             if file_name is None:
    #                 file_name = f'qst{n}'
    #         elif method == 'ts':
    #             extra = f' opt(ts,noeigentest,calcfc,maxcycles=1000) freq ' + extra + ' '
    #             if file_name is None:
    #                 file_name = 'ts'
    #         else:
    #             print ('Unknown TS searching method was used !!!')
    #             return None, {}
    #     #print (ts_molecule)
    #     self.make_input(molecules,chg,multiplicity,file_name=file_name,extra=extra)
    #     os.system(f'{self.command} {file_name}.com')
    #     
    #     # Read output
    #     p = cclib.parser.Gaussian(f'{file_name}.log')
    #     data = p.parse()
    #     self.move_file(file_name,save_directory)
    #     # Get minimal energy geometry
    #     try:
    #         index = np.argmin(data.scfenergies)
    #     except:
    #         print ('TS Calculation did not start properly ...')
    #         os.chdir(current_directory)
    #         return None, {}
    #     if index > len(data.atomcoords) - 1:
    #         index = -1
    #     coordinate_list = data.atomcoords[index] # Sometimes length is not matched, it performs extra scfenergy calculation
    #     converter = 1
    #     if self.energy_unit == 'kcal':
    #         converter = 23.06
    #     if self.energy_unit == 'Hartree':
    #         converter = 0.036749326681
    #         #converter = 1/27.2114
    #     energy = data.scfenergies[index] * converter
    #     process.locate_molecule(ts_molecule,coordinate_list,False)
    #     ts_molecule.energy = energy
    #     if not check_frequency:
    #         return ts_molecule,{} 
    #     # Get frequency ...
    #     imaginary_vibrations = dict()
    #     try:
    #         vibmodes = data.vibdisps
    #         vibfreqs = data.vibfreqs
    #     except:
    #         print ('TS Calculation not well converged ...')
    #         os.chdir(current_directory)
    #         return ts_molecule, {}
    #     m = len(vibfreqs)
    #     for i in range(m):
    #         freq = vibfreqs[i]
    #         mode = vibmodes[i]
    #         freq = round(freq,4)
    #         if freq < 0:
    #             imaginary_vibrations[freq] = mode
    #
    #     #os.system('mv new.chk old.chk')
    #     os.chdir(current_directory)
    #     
    #     return ts_molecule,imaginary_vibrations

    # def run_irc(self,ts_molecule,chg=None,multiplicity=None,file_name='irc',extra = '',save_directory=None,chkpoint_file=''):
    #     original_content = self.content
    #     if chkpoint_file != '':
    #         self.content = original_content + f'%chk={chkpoint_file}\n'
    #     current_directory = os.getcwd()
    #     os.chdir(self.working_directory)
    #     
    #     if chkpoint_file == '':
    #         if 'irc' not in extra:
    #             extra = extra + f' irc(LQA, recorrect=never, CalcFC, maxpoints=60, StepSize=15, maxcycles=100) '
    #     else:
    #         if 'irc' not in extra:
    #             extra = extra + f' irc(LQA, recorrect=never, maxpoints=60, StepSize=15, maxcycles=100) '
    #     
    #     self.make_input([None,None,ts_molecule],chg,multiplicity,file_name=file_name,extra=extra)
    #     self.content = original_content
    #     os.system(f'{self.command} {file_name}.com')
    #     
    #     # Read output
    #     p = cclib.parser.Gaussian(f'{file_name}.log')
    #     data = p.parse()
    #     self.move_file(file_name,save_directory)
    #
    #     irc_trajectory = []
    #     try:
    #         geometries = data.atomcoords
    #         energies = data.scfenergies
    #     except:
    #         print ('IRC Calculation did not start properly !!!')
    #         os.chdir(current_directory)
    #         return irc_trajectory
    #
    #     converter = 1
    #     if self.energy_unit == 'kcal':
    #         converter = 23.06
    #     if self.energy_unit == 'Hartree':
    #         converter = 0.036749326681
    #     energies *= converter
    #
    #     for i,coordinate_list in enumerate(geometries):
    #         copied_molecule = ts_molecule.copy()
    #         process.locate_molecule(copied_molecule,coordinate_list,True)
    #         copied_molecule.energy = energies[i]
    #         irc_trajectory.append(copied_molecule)
    #
    #     #os.system('mv new.chk old.chk')
    #     os.chdir(current_directory)        
    #     return irc_trajectory

    def clean_scratch(self,file_name='test.mop'):
        working_directory = self.working_directory
        chk_directory = os.path.join(working_directory,'*.arc')
        os.system(f'rm {chk_directory}')
        file_directory = os.path.join(working_directory,'test.mop')
        os.system(f'rm {file_directory}')
        file_directory = os.path.join(working_directory,'test.log')
        os.system(f'rm {file_directory}')
        file_directory = os.path.join(working_directory,'test.out')
        os.system(f'rm {file_directory}')

if __name__ == "__main__":
    mopac = Mopac()
    mol = chem.Molecule("O")
    conformer = mol.sample_conformers(10)[0]
    mopac.optimize_geometry(conformer)
    print(conformer.energy)
    conformer.print_coordinate_list()
    mopac.optimize_geometry_with_contraints(conformer, 
                                            constraints={(0, 1): 2.0,
                                                         (0, 2): 2.0},
                                            file_name='test2')
    # print(conformer.energy)
    conformer.print_coordinate_list()
