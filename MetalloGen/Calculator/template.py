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


### Module for reading gaussian files ###
import cclib

### ace-reaction libraries ###
import chem
import process

'''
You can define your own calculator, depending on your using software!!!
Here, you must note that below functions must be defined in order to run calculations with MCD: __init__, get_energy, get_force, relax_geometry
See orca.py and gaussian.py as examples. How they 
You can also try making calculations using ase. In our case, we did not use ASE modules due to some optimizing issues ...
'''
class Calculator:
    
    def __init__(self):
        self.working_directory = os.getcwd()
        self.energy_unit = 'Hartree'

    def load_content(self,template_directory):
        pass

    def load_basis(self,basis_directory):
        pass        

    def change_working_directory(self,working_directory):
        # Get current reaction coordinate
        if not os.path.exists(working_directory):
            print ('Working directory does not exist! Creating new directory ...')
            os.system(f'mkdir {self.working_directory}')
            self.working_directory = working_directory

    def get_content(self):
        return self.content 

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


    def get_energy(self,molecule,chg=None,multiplicity = None):
        '''
        Must return energy with desired unit defined in the Calculator
        '''
        pass

    
    def get_force(self,molecule,chg=None,multiplicity=None):
        '''
        Must return force with desired unit defined in the Calculator
        '''
        pass

    def optimize_geometry(self,molecule,constraints={},chg=None,multiplicity=None,file_name='test',extra='',return_data = False):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        self.make_input(molecule,chg,multiplicity,'opt',file_name,constraints,extra)
        os.system(f'{self.command} {file_name}.com')
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
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
        #os.system('mv new.chk old.chk')
        os.chdir(current_directory)
        if return_data:
            return data


    def make_input(self,molecule,chg=None,multiplicity=None,option='sp',file_name='test',constraints={},extra=''):
        f = open(f'{file_name}.com','w')
        if chg is None or multiplicity is None:
            chg, multiplicity = self.get_default_mol_params(molecule)
        content = self.get_content()
        content = content + option + f'{extra}\n\ntest\n\n'
        f.write(content)
        self.write_molecule_info(f,molecule,chg,multiplicity)
        ### Write constraints
        for constraint in constraints:
            constraint_info = [] 
            if len(constraint) == 2:
                constraint_info.append('B')
            elif len(constraint) == 3:
                constraint_info.append('A')
            else:
                constraint_info.append('D') 
            for index in constraint:
                constraint_info.append(str(index+1))
            constraint_info.append('F')
            f.write(' '.join(constraint_info)+'\n')
        f.write('\n') # Additional empty line required
        self.write_basis(f)
        f.write('\n')
        f.close()

    def relax_geometry(self):
        '''
        '''
        pass
