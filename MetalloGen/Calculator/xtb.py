### python provided modules ###
import time
import os
import sys
from copy import deepcopy
import subprocess
import pickle
import datetime
import argparse
import distutils.spawn

import numpy as np

import cclib
### ace-reaction libraries ###
from MetalloGen import chem, process



def parse_energy(directory): # Read {file_name}.log
    try:
        with open(directory,'r') as f:
            lines = f.readlines()
    except:
        return None
    if '.energy' in directory:
       energy_line = lines[1].strip().split()
       energy = float(energy_line[-1])
       return energy  
    else:
        temp = 100000
        for i, line in enumerate(lines):
            if 'FINAL SINGLE' in line:
                energy_line = line.strip().split()
                energy = float(energy_line[-1])
                return energy
        return None

def parse_force(directory): # Read {file_name}.engrad
    with open(directory) as f:
        lines = f.readlines()

    temp = 100000
    grads = []
    energy = None
    for i, line in enumerate(lines):
        if '# The current total energy' in line:
            energy = float(lines[i+2].strip())
        if '# The current gradient in Eh/bohr' in line:
            temp = i
        if i > temp+1:
            if "#" in line:
                break
            grads.append(float(line.strip()))
    try:
        n = len(grads)
    except:
        return None,None

    if n == 0:
        return None,None

    if n%3 != 0:
        print ('what?')
        return None,None

    n = int(n/3)
    force = -np.array(grads) # F = -g
    force = np.reshape(force,(n,3))
    return energy,force


def parse_hessian(directory): # Read {file_name}.hess
    # Find $hessian
    index = 0
    try:
        with open(directory,'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if '$hessian' in line:
                    index = idx
                    index += 1
                    break
    except:
        return None
    flatten_hessian = []
    for i in range(index,len(lines)):
        values = lines[i].strip().split()
        values = [float(value) for value in values]
        flatten_hessian += values
    n = int(np.sqrt(len(flatten_hessian)))
    hessian = np.reshape(flatten_hessian,(n,n))
    #print (hessian)
    #d, v = np.linalg.eig(hessian)
    #print (np.sort(d))
    return hessian


def parse_vibrations(directory):
    target_line = 0 
    # Get frequency
    try:
        with open(directory,'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if '$vibrational' in line:
                    index = idx
                    index += 1
                    break
    except:
        return None
    # Read frequency
    n = int(lines[index])
    index += 1
    frequencies = []
    vibrations = np.zeros((n,n))
    for i in range(n):
        frequency_info = lines[index].strip().split()
        frequencies.append(float(frequency_info[-1]))
        index += 1

    # Read normal mode
    while True:
        line = lines[index]
        if '$normal_modes' in line:
            break
        index += 1
    index += 2
    cnt = 0
    while cnt < n:
        index_infos = lines[index].strip().split()
        index += 1
        for i in range(n):
            values = lines[index].strip().split()
            index += 1
            for j,index_info in enumerate(index_infos):
                index_info = int(index_info)
                vibrations[i][index_info] = float(values[j+1])
        cnt += len(index_infos) 

    return frequencies,vibrations

def parse_opt(directory): # Read {file_name}.trj.xyz

    # Open ORCA output file and read its contents
    try:
        with open(directory) as f:
            lines = f.readlines()
    except:
        return []

    trajectory = []
    index = 0
    while index < len(lines):
        num_atom = int(lines[index].strip()) # read num_atom
        index += 1
        infos = lines[index].strip().split() # Read energy
        energy = float(infos[-1])
        index += 1
        atom_list = []
        for i in range(num_atom):
            atom_info = lines[index].strip().split()
            index += 1
            element = atom_info[0]
            x = float(atom_info[1])
            y = float(atom_info[2])
            z = float(atom_info[3])
            atom = chem.Atom(element)
            atom.x = x
            atom.y = y
            atom.z = z
            atom_list.append(atom)
        if len(atom_list) > 0:
            molecule = chem.Molecule()
            molecule.atom_list = atom_list
            molecule.energy = energy
            trajectory.append(molecule)
        else:
            break
    return trajectory



class XTB:

    
    def __init__(self,command='orca',working_directory = None):
        # Check xtb
        check = distutils.spawn.find_executable('xtb')
        self.name = 'xtb'
        if check is None:
            print ('xtb not found!')
            exit()
 
        # Also, check orca
        check = distutils.spawn.find_executable(command)
        if check is None:
            print ('orca not found!')
            exit()
        
        self.command = command
        self.nproc=1
        self.content = '! XTB2 '
        self.energy_unit = 'Hartree'
        #self.distance_unit = 'Bohr'
        #self.radian_unit = 'Radian'

        if working_directory is not None:
            if not os.path.exists(working_directory):
                os.system(f'mkdir {working_directory}')
                if not os.path.exists(working_directory):
                    print ('Defined working directory does not exist!!!')
                    working_directory = None

        if working_directory is None:
            working_directory = os.getcwd()
            print ('working directory is set to current diretory:',working_directory)

        self.working_directory = working_directory
        self.error_directory = None

    def __str__(self):
        content = ''
        content = content + f'working_directory: {self.working_directory}\n'
        content = content + f'command: {self.command}\n'
        content = content + f'Energy: {self.energy_unit}\n\n'
        content = content + f'###### qc_input ######\n{self.content}\n'
        return content

    def set_error_directory(self,error_directory):
        try:
            if os.path.exists(error_directory):
                self.error_directory = error_directory
            else:
                print (f'Given error directory: {error_directory} does not exist!!! ')
        except:
            print (error_directory)
            print ('Check your error directory !!!')

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

    def load_content(self,template_directory):
        content = ''
        with open(template_directory) as f:
            for line in f:
                content = content + line
        self.content = content


    def get_default_mol_params(self,molecule):
        try:
            chg = molecule.get_chg()
        except:
            chg = 0
        try:
            e_list = molecule.get_num_of_lone_pair_list()
            num_of_unpaired_e = len(np.where((2*e_list) % 2 == 1)[0])
            multiplicity = num_of_unpaired_e + 1
        except:
            z_sum = np.sum(molecule.get_z_list())
            multiplicity = (z_sum - chg) % 2 + 1
        return chg,multiplicity

    def make_input(self,molecules,chg, multiplicity, file_name='test',constraints={},extra=''):

        if chg is None or multiplicity is None:
            if molecules[0] is not None:
                molecule = molecules[0]
                chg, multiplicity = self.get_default_mol_params(molecules[0])
            elif molecules[-1] is not None:
                molecule = molecules[-1]
                chg, multiplicity = self.get_default_mol_params(molecules[-1])

        inpstring = self.content.strip()
        if inpstring[0] == '#':
            inpstring = inpstring[1:]
        inpstring = inpstring + extra
        
        if len(molecules[0].atom_list) == 1:
            inpstring = inpstring.replace('opt','')
            inpstring = inpstring.replace('OPT','')

        inpstring += f'\n*xyz {chg} {multiplicity}\n'
        for atom in molecules[0].atom_list:
            inpstring += atom.get_content()

        inpstring += '*'
        with open(f'{file_name}.com', 'w') as f:
            f.write(inpstring)

    def move_file(self,file_name,save_directory):
        if file_name is not None and save_directory is not None:
            os.system(f'mv {file_name}.* {save_directory}/')
            
    
    def run(self,molecule,chg=None,multiplicity=None,file_name='test',save_directory=None): # Running with user defined result and return raw data
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        self.make_input([molecule],chg,multiplicity,file_name,{},'')
        #print(os.path.isfile(f'{file_name}.com'))
        os.system(f'{self.command} {file_name}.com > {file_name}.log')
        if len(molecule.atom_list) > 1 and 'opt' in str.lower(self.content):
            try:
                relaxing_path = parse_opt(os.path.join(self.working_directory,f'{file_name}_trj.xyz'))
            except:
                relaxing_path = None
                print('Orca Optimization Calculation failed !!!')
            if len(relaxing_path) < 2:
                if self.error_directory is not None:
                    file_directory = os.path.join(self.error_directory,'orca.err')
                    with open(file_directory,'w') as f:
                        f.write('Orca Optimization Calculation failed !!!\n')
                        name = f'{file_name}.log'
                        f.write(f'Check {os.path.join(self.working_directory,name)} ...\n')
            process.locate_molecule(molecule,relaxing_path[-1].get_coordinate_list())
        energy = parse_energy(os.path.join(self.working_directory,f'{file_name}.energy'))
        molecule.energy = energy
        os.chdir(current_directory)
        if energy is not None:
            return True
        else: return False

    def get_energy(self,molecule,chg=None,multiplicity = None,file_name='sp',extra='',save_directory=None):
        '''
        Must return energy with desired unit defined in the Calculator
        '''
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        self.make_input([molecule],chg,multiplicity,file_name=file_name,extra=extra)
        os.system(f'{self.command} {file_name}.com > {file_name}.log')
        energy = parse_energy(os.path.join(self.working_directory,f'{file_name}.energy'))
        self.move_file(file_name,save_directory)
        converter = 1
        if energy is None:
            if self.error_directory is not None:
                file_directory = os.path.join(self.error_directory,'orca.err')
                with open(file_directory,'w') as f:
                    f.write('Orca Energy Calculation failed !!!\n')
                    name = f'{file_name}.log'
                    f.write(f'Check {os.path.join(self.working_directory,name)} ...\n')
            return None
        if self.energy_unit == 'kcal':
            converter = 627.509
        if self.energy_unit == 'eV':
            converter = 27.2114079527
        os.chdir(current_directory)
        return converter*energy


    def get_force(self,molecule,chg=None,multiplicity=None,file_name='force',extra='',save_directory=None):
        '''
        Must return force with desired unit defined in the Calculator
        '''
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        if 'engrad' not in extra:
            extra = ' engrad ' + extra
        self.make_input([molecule],chg,multiplicity,file_name=file_name,extra=extra)
        os.system(f'{self.command} {file_name}.com > {file_name}.log')
        self.move_file(file_name,save_directory)
        energy,force = parse_force(os.path.join(self.working_directory,f'{file_name}.engrad'))
        bohr_to_angstrom = 0.529177 # Units are Hartree/bohr in chkpoint file
        if force is None:
            if self.error_directory is not None:
                file_directory = os.path.join(self.error_directory,'orca.err')
                with open(file_directory,'w') as f:
                    f.write('Orca Force Calculation failed !!!\n')
                    name = f'{file_name}.log'
                    f.write(f'Check {os.path.join(self.working_directory,name)} ...\n')
            return None
        os.chdir(current_directory)
        return force/bohr_to_angstrom

    def get_hessian(self,molecule,chg=None,multiplicity=None,file_name='hessian',extra='',save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        original_content = self.content
        self.make_input([molecule],chg,multiplicity,file_name=file_name,extra=extra)
        os.system(f'{self.command} {file_name}.com > {file_name}.log')
        bohr_to_angstrom = 0.529177 # Units are Hartree/bohr in chkpoint file
        if force is None:
            if self.error_directory is not None:
                file_directory = os.path.join(self.error_directory,'orca.err')
                with open(file_directory,'w') as f:
                    f.write('Orca Force Calculation failed !!!\n')
                    name = f'{file_name}.log'
                    f.write(f'Check {os.path.join(self.working_directory,name)} ...\n')
            return None,None
        if hessian is None:
            if self.error_directory is not None:
                file_directory = os.path.join(self.error_directory,'orca.err')
                with open(file_directory,'w') as f:
                    f.write('Orca Hessian Calculation failed !!!\n')
                    name = f'{file_name}.log'
                    f.write(f'Check {os.path.join(self.working_directory,name)} ...\n')
            return None,None
        self.move_file(file_name,save_directory)
        os.chdir(current_directory)
        return None, hessian/bohr_to_angstrom**2


    def optimize_geometry(self,molecule,constraints={},chg=None,multiplicity=None,file_name='opt',extra='',save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        if 'opt' not in extra:
            extra = ' opt ' + extra
        
        # Additional constraint
        if len(constraints) > 0:
            if 'geom' not in extra:
                extra = extra + '\n\n%geom\nConstraints\n'
            else:
                extra = extra + 'Constraints\n'
            for constraint in constraints:
                constraint_info = []
                if len(constraint) == 2:
                    constraint_info.append('{B')
                elif len(constraint) == 3:
                    constraint_info.append('{A')
                else:
                    constraint_info.append('{D')
                for index in constraint:
                    constraint_info.append(str(index))
                constraint_info.append('C}')
                extra +=  (' '.join(constraint_info)+'\n')
            extra += 'end\n'
        if 'geom' in extra:
            extra += 'end\n'

        self.make_input([molecule],chg,multiplicity,file_name=file_name,extra=extra,constraints=constraints)
        os.system(f'{self.command} {file_name}.com > {file_name}.log')
        relaxing_path = parse_opt(os.path.join(self.working_directory,f'{file_name}_trj.xyz'))
        if len(relaxing_path) < 2:
            if self.error_directory is not None:
                file_directory = os.path.join(self.error_directory,'orca.err')
                with open(file_directory,'w') as f:
                    f.write('Orca Optimization Calculation failed !!!\n')
                    name = f'{file_name}.log'
                    f.write(f'Check {os.path.join(self.working_directory,name)} ...\n')
            return []
        energy = parse_energy(os.path.join(self.working_directory,f'{file_name}.energy'))
        process.locate_molecule(molecule,relaxing_path[-1].get_coordinate_list())
        molecule.energy = energy
        os.chdir(current_directory)
        return relaxing_path

    def relax_geometry(self,molecule,constraints,chg=None,multiplicity=None,file_name='opt',num_relaxation=5,maximal_displacement=1000,extra='',save_directory=None):
        '''
        '''
        extra = '\n%geom\n'
        if maximal_displacement < 1000:
            maximal_displacement = -abs(maximal_displacement)
            extra = extra + f'Trust {maximal_displacement}\n'
        if num_relaxation is not None and num_relaxation > 0:
            extra = extra + f'MaxIter {num_relaxation}\n'
        return self.optimize_geometry(molecule,constraints,chg,multiplicity,file_name,extra,save_directory)


    def search_ts(self,molecules,chg=None,multiplicity=None,method='ts',file_name=None,extra = '',save_directory=None,check_frequency = True):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        n = 0
        for molecule in molecules:
            if molecule is not None:
                n += 1
        if n == 1:
            ts_molecule = molecules[-1].copy()
        else:
            ts_molecule = molecules[0].copy()
        if method == 'ts':
            extra = ' optts '
            self.make_input([ts_molecule],chg,multiplicity,file_name=file_name,extra=extra)
        else:
            # Run NEB method
            # Prepare reactant and product
            # Write NEB input
            # Perform NEB
            pass
        os.system(f'{self.command} {file_name}.com > {file_name}.log')
        
        # Read output
        trajectory = parse_opt(f'{file_name}_trj.xyz')
        energy = parse_energy(os.path.join(self.working_directory,f'{file_name}.energy'))
        process.locate_molecule(ts_molecule,trajectory[-1].get_coordinate_list())
        ts_molecule.energy = energy
        if not check_frequency:
            return ts_molecule,{} 
        
        # Second, compute hessian to get vibration ...
        imaginary_vibrations = dict()
        # Make new xyz file ...
        ts_molecule.write_geometry('ts.xyz')
        chg, multiplicity = self.get_default_mol_params(ts_molecule)
        xtb_content = f'xtb -hess ts.xyz -c {chg} -u {multiplicity-1}'
        # Include solvent, if exists ...
        os.system() 
        try:
            vibfreqs, vibmodes = parse_vibrations(os.path.join(self.working_directory,f'{file_name}.hess'))
        except:
            print ('TS Calculation not well converged ...')
            os.chdir(current_directory)
            return ts_molecule, {}
        m = len(vibfreqs)
        for i in range(m):
            freq = vibfreqs[i]
            mode = np.reshape(vibmodes[i],(m/3,3))
            freq = round(freq,4)
            if freq < 0:
                imaginary_vibrations[freq] = mode
        os.chdir(current_directory)
        
        return ts_molecule,imaginary_vibrations


if __name__ == '__main__':
    import sys
    parse_hessian(sys.argv[1])
