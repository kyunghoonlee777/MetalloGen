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
import shutil


### Module for reading gaussian files ###
import cclib

### ace-reaction libraries ###
import chem
import process

# Ugly fix for cclib error which occurs when 
# only two rotational constants are given.
# Replaces *************** to an arbitrary value
def cclib_rot_const_fix(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace the specified string
    updated_content = content.replace(
    "Rotational constants (GHZ):***************", 
    "Rotational constants (GHZ):0.0000000000001"
    )

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)

def remove_option(removing_option, options):
    index = 0
    while index < len(options):
        option = options[index]
        lower = str.lower(option)
        if lower == removing_option:
            options.remove(option)
        else:
            index += 1            

def copy_params(params):
    pass
    
class Gaussian:

    def __init__(self, command="g16", working_directory=None):
        # Check command
        check = distutils.spawn.find_executable(command)
        if check is not None:
            self.command = command
        else:
            # Check default command
            commands = ["g09", "g16"]
            found = False
            for old_command in commands:
                check = distutils.spawn.find_executable(old_command)
                if check is not None:
                    self.command = old_command
                    print(
                        f"command {self.command} is used for running Gaussian, instead of {command}!"
                    )
                    found = True
                    break
            if not found:
                print("Gaussian not found!")
                exit()

        if working_directory is not None:
            if not os.path.exists(working_directory):
                os.system(f"mkdir {working_directory}")
                if not os.path.exists(working_directory):
                    print("Defined working directory does not exist!!!")
                    working_directory = None

        if working_directory is None:
            working_directory = os.getcwd()
            print("working directory is set to current diretory:", working_directory)

        self.working_directory = working_directory
        # Check save directory
        self.content = "#p pm6 "
        self.energy_unit = "Hartree"
        self.basis = ""
        self.nomicro = False
        self.error_directory = None
        self.rm_command = '/bin/rm '


    def __str__(self):
        content = ""
        content = content + f"working_directory: {self.working_directory}\n"
        content = content + f"command: {self.command}\n"
        content = content + f"Energy: {self.energy_unit}\n\n"
        content = content + f"###### qc_input ######\n{self.content}\n"
        return content

    def load_content(self, template_directory):
        if os.path.exists(template_directory):
            content = ""
            with open(template_directory) as f:
                lines = f.readlines()
                n = len(lines)
                for i, line in enumerate(lines):
                    if i == n - 1:
                        line = line.strip()
                    content = content + line
            self.content = content
        else:
            print("qc input file does not exist!!!")

    def load_basis(self, basis_directory):
        if os.path.exists(basis_directory):
            basis = ""
            with open(basis_directory) as f:
                for line in f:
                    basis = basis + line
            self.basis = basis
        else:
            print("basis file does not exist!!!")

    def change_working_directory(self, working_directory):
        # Get current reaction coordinate
        if not os.path.exists(working_directory):
            print("Working directory does not exist! Creating new directory ...")
            os.system(f"mkdir {working_directory}")
            if not os.path.exists(working_directory):
                print(
                    "Cannot create working directory!!!\n Recheck your working directory ..."
                )
                exit()
            else:
                print("working directory changed into:", working_directory)
                self.working_directory = working_directory
        else:
            print("working directory changed into:", working_directory)
            self.working_directory = working_directory
        # os.environ['GAUSS_SCRDIR'] = working_directory

    def execute(self,file_name,save_chk_file=None):
        os.system(f"{self.command} {file_name}.com")
        cclib_rot_const_fix(f"{file_name}.log")
        # Save chk file if exists ...
        if save_chk_file:
            os.system(f'mv abcdefg.chk {save_chk_file}')


    def get_content(self):
        content = self.content
        return content

    def get_default_mol_params(self, molecule):
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
                num_of_unpaired_e = len(np.where((2 * e_list) % 2 == 1)[0])
                multiplicity = num_of_unpaired_e + 1
            except:
                z_sum = np.sum(molecule.get_z_list())
                multiplicity = (z_sum - chg) % 2 + 1
        return chg, multiplicity

    def write_molecule_info(self, f, molecule, chg, multiplicity):
        f.write(f"{int(chg)} {int(multiplicity)}\n")
        for atom in molecule.atom_list:
            f.write(atom.get_content())
        f.write("\n")

    def write_basis(self, f):
        if self.basis != "":
            f.write(f"{self.basis}\n")

    def switch_to_xtb_gaussian(self):
        if not shutil.which('xtb'): 
            return False
        try:
            xtbbin = os.environ['xtbbin']
        except:
            return False
        xtb_directory = '\'' + os.environ['xtbbin']+'\' '
        self.content = f'#p external={xtb_directory} /sdd '
        self.nomicro = True
        return True
        

    # TODO: This will be handled very soon ...
    def make_uff_input(
        self,
        molecule,
        chg=None,
        multiplicity=None,
        option="opt",
        file_name="test",
        constraints=[],
    ):
        bo_matrix = molecule.get_bo_matrix()
        if bo_matrix is None:
            bo_matrix = molecule.get_adj_matrix()
            if bo_matrix is None:
                bo_matrix = process.get_adj_matrix_from_distance(molecule, 1.2)
                if bo_matrix is None:
                    print("Cannot define connectivity !!!")
                    return None
        # bo matrix is now defined ...
        f = open(f"{file_name}.com", "w")
        if chg is None or multiplicity is None:
            chg, multiplicity = self.get_default_mol_params(molecule)
        content = ""
        content = content + option + "\n\ntest\n\n"
        f.write(content)
        self.write_molecule_info(f, molecule, chg, multiplicity)
        ### Write constraints
        for constraint in constraints:
            constraint_info = []
            if len(constraint) == 1:
                constraint_info.append("X")
            elif len(constraint) == 2:
                constraint_info.append("B")
            elif len(constraint) == 3:
                constraint_info.append("A")
            else:
                constraint_info.append("D")
            for index in constraint:
                constraint_info.append(str(index + 1))
            constraint_info.append("F")
            f.write(" ".join(constraint_info) + "\n")
        f.write("\n")  # Additional empty line required
        self.write_basis(f)
        f.write("\n")
        f.close()


    def make_input(
        self,
        molecules,
        chg=None,
        multiplicity=None,
        file_name="test",
        params = dict(),
        **kwargs
    ):
        f = open(f"{file_name}.com", "w")
        if chg is None or multiplicity is None:
            if molecules[0] is not None:
                chg, multiplicity = self.get_default_mol_params(molecules[0])
            elif molecules[-1] is not None:
                chg, multiplicity = self.get_default_mol_params(molecules[-1])
        content = self.get_content()
        # Check chk file ...
        if 'chk_file' in kwargs and kwargs['chk_file'] is not None:
            value = kwargs['chk_file']
            content = f'%oldchk={value}\n' + content

        if 'save_chk_file' in kwargs and kwargs['save_chk_file'] is not None:
            value = kwargs['save_chk_file']
            content = f'%chk=abcdefg.chk\n' + content

        # If % in key, it is the top part of the content ...
        for key, value in params.items(): 
            if type(value) is str: # Or check value ...
                content = f'{key}={value}\n' + content
            else:
                # It should be list ...
                if key != 'constraint':
                    # Remove repeating words ...
                    # TODO: This should be done more precisely ...
                    final_value = []
                    for option in value:
                        lower = str.lower(option)
                        if lower in final_value:
                            continue
                        else:
                            final_value.append(lower)
                    if len(final_value) > 0:
                        phrase = ','.join(final_value)
                        content = content + f'{key}({phrase})\n'
                    else:
                        content = content + f'{key}\n'
        
        f.write(content.strip()+'\n\n')
        labels = ["R", "P", "TS"]
        for i, molecule in enumerate(molecules):
            if molecule is not None:
                f.write(f"{labels[i]}\n\n")
                self.write_molecule_info(f, molecule, chg, multiplicity)

        ### Write constraints
        if 'constraint' in params:
            for constraint in params['constraint']:
                constraint_info = []
                if len(constraint) == 1:
                    constraint_info.append("X")
                elif len(constraint) == 2:
                    constraint_info.append("B")
                elif len(constraint) == 3:
                    constraint_info.append("A")
                else:
                    constraint_info.append("D")
                for index in constraint:
                    constraint_info.append(str(index + 1))
                constraint_info.append("F")
                f.write(" ".join(constraint_info) + "\n")
            f.write("\n")  # Additional empty line required
        self.write_basis(f)
        f.write("\n")
        f.close()

    def move_file(self, file_name, save_directory):
        if file_name is not None and save_directory is not None:
            file_directory = f"{file_name}.com"
            os.system(f"mv {file_name}.com {save_directory}/")
            os.system(f"mv {file_name}.log {save_directory}/")
            #os.system(f"mv *chk {save_directory}/") # Directly specified as kwargs ...

    # TODO: This part should be more precise later ...
    def run(
        self,
        molecules,
        chg=None,
        multiplicity=None,
        file_name="test",
        params=dict(),
        save_directory=None,
        **kwargs
    ):  # Running with user defined result and return raw data
        os.chdir(self.working_directory)
        self.make_input(molecules, chg, multiplicity, file_name, params,**kwargs)
        if 'save_chk_file' in kwargs and kwargs['save_chk_file'] is not None:
            save_chk_file = kwargs['save_chk_file']
        else:
            save_chk_file = None
        self.execute(file_name,save_chk_file)
        p = cclib.parser.Gaussian(f"{file_name}.log")
        data = p.parse()
        return data


    def get_energy(
        self,
        molecule,
        chg=None,
        multiplicity=None,
        file_name="sp",
        params=dict(),
        save_directory=None,
        **kwargs
    ):
        current_directory = os.getcwd()
        params = deepcopy(params)
        if 'sp' not in params:
            params['sp'] = []
        data = self.run([molecule],chg,multiplicity,file_name,params,save_directory,**kwargs)
        converter = 1
        if self.energy_unit == "kcal":
            converter = 23.06
        if self.energy_unit == "Hartree":
            converter = 0.036749326681
        # os.system('mv new.chk old.chk')
        os.system(f'{self.rm_command} Gau-*')
        self.move_file(file_name,save_directory)
        os.chdir(current_directory)
        # print("is data empty??", data, "\n", dir(data))
        try:
            data.scfenergies[-1]
        except:
            return None
        return converter * data.scfenergies[-1]


    def get_force(
        self,
        molecule,
        chg=None,
        multiplicity=None,
        file_name="force",
        params=dict(),
        save_directory=None,
        **kwargs
    ):
        current_directory = os.getcwd()
        params = deepcopy(params)
        if 'force' not in params:
            params['force'] = []
        if 'Symmetry' not in params:
            params['Symmetry'] = ['None']
        data = self.run([molecule],chg,multiplicity,file_name,params,save_directory,**kwargs)
        # os.system('mv new.chk old.chk')
        self.move_file(file_name,save_directory)
        os.chdir(current_directory)
        try:
            data.grads[-1] # ccilb grads is the negative of gradient of the energy
        except:
            return None
        return data.grads[-1]


    def get_hessian(
        self,
        molecule,
        chg=None,
        multiplicity=None,
        file_name='hessian',
        params = dict(),
        save_directory=None,
        **kwargs
    ):
        current_directory = os.getcwd()
        params = deepcopy(params)
        if 'freq' not in params:
            params['freq'] = []
        if 'Symmetry' not in params:
            params['Symmetry'] = ['None']
        if 'save_chk_file' in kwargs and kwargs['save_chk_file'] is not None:
            save_chk_file = kwargs['save_chk_file']
        else:
            save_chk_file = 'hessian.chk'
            kwargs['save_chk_file'] = save_chk_file
        data = self.run([molecule],chg,multiplicity,file_name,params,save_directory,**kwargs)
        os.system(f'formchk {save_chk_file}')
        force = []
        hessian = []
        bohr_to_angstrom = 0.529177 # Units are Hartree/bohr in chkpoint file
        fchk_file = f'{save_chk_file[:-4]}.fchk'
        #print ('hhhhhhhhhhhhhhhhhhhhhhh',os.getcwd(),fchk_file)
        if not os.path.exists(fchk_file):
            if self.error_directory is not None:
                file_directory = os.path.join(self.error_directory,'gaussian.err')
                with open(file_directory,'w') as f:
                    f.write('Gaussian Hessian Calculation failed !!!\n')
                    name = f'{file_name}.log'
                    f.write(f'Check {os.path.join(self.working_directory,name)} ...\n')
            return None

        # Read fchk file
        with open(fchk_file,'r') as f:
            lines = f.readlines()
            index = 0
            while index < len(lines):
                line = lines[index]
                if 'Cartesian Gradient' in line: # Real gradient ...
                    index += 1
                    line = lines[index]
                    infos = line.strip().split()
                    # Read line
                    while True:
                        try:
                            float(infos[0])
                        except:
                            break
                        force = force + [float(info) for info in infos]
                        index += 1
                        line = lines[index]
                        infos = line.strip().split()

                if 'Cartesian Force Constant' in line:
                    index += 1
                    line = lines[index]
                    infos = line.strip().split()
                    # Read line
                    while True:
                        try:
                            float(infos[0])
                        except:
                            break
                        hessian = hessian + [float(info) for info in infos]
                        index += 1
                        line = lines[index]
                        infos = line.strip().split()

                index += 1
                if len(force) > 0 and len(hessian) > 0:
                    break
                
        n = len(molecule.atom_list)
        force = -np.array(force)
        force = np.reshape(force,(n,3))
        new_hessian = np.zeros((3*n,3*n))
        cnt = 0
        for i in range(3*n):
            for j in range(i+1):
                new_hessian[i][j] = new_hessian[j][i] = hessian[cnt]
                cnt += 1
        hessian = new_hessian
        self.move_file(file_name,save_directory)
        os.chdir(current_directory)
        return force/bohr_to_angstrom,hessian/bohr_to_angstrom**2


    def optimize_geometry(
        self,
        molecule,
        constraints=[],
        chg=None,
        multiplicity=None,
        file_name="opt",
        params=dict(),
        save_directory=None,
        **kwargs
    ):
        current_directory = os.getcwd()

        params = deepcopy(params)

        if 'opt' not in params:
            params['opt'] = []

        # Check initial hessian        
        if 'initial_hessian' in kwargs: # calcfc, readfc
            initial_hessian = kwargs['initial_hessian']
        else:
            initial_hessian = None

        if initial_hessian is not None:
            # Remove hessian related keywords ...
            params['opt'].append(initial_hessian)

        if len(constraints) > 0:
            params['opt'].append('modredundant')
            params['constraint'] = constraints
 
        if self.nomicro:
            params['opt'].append('nomicro')

        #print (params)               
        data = self.run([molecule],chg,multiplicity,file_name,params,save_directory,**kwargs)
        self.move_file(file_name,save_directory)
        os.chdir(current_directory)

        try:
            data.scfenergies
        except:
            return None

        # Get minimal energy geometry
        index = -1 # TODO: Maybe use the lowest energy ???
        coordinate_list = data.atomcoords[index]  # Sometimes length is not matched, it performs extra scfenergy calculation
        converter = 1
        if self.energy_unit == "kcal":
            converter = 23.06
        if self.energy_unit == "Hartree":
            converter = 0.036749326681
            # converter = 1/27.2114
        energy = data.scfenergies[index] * converter
        process.locate_molecule(molecule, coordinate_list, False)
        molecule.energy = energy

        # os.system('mv new.chk old.chk')

    def relax_geometry(
        self,
        molecule,
        constraints=[],
        chg=None,
        multiplicity=None,
        file_name="test",
        num_relaxation=5,
        maximal_displacement=1000,
        params = dict(),
        save_directory=None,
        **kwargs
    ):
        
        params = deepcopy(params)

        # Reset params (more closely, opt option)
        params['opt'] = ['loose']
        params['opt'].append(f'maxcycles={num_relaxation}')
        params['opt'].append(f'notrust')
        params['Symmetry'] = ['None']
 
        if maximal_displacement < 100:
            max_step = int(maximal_displacement * 100) + 1
            params['opt'].append(f'maxstep={max_step}')
        if 'algorithm' in kwargs:
            algorithm = kwargs['algorithm']
            params['opt'].append(params)

        self.optimize_geometry(
            molecule, constraints, chg, multiplicity, file_name, params, save_directory, **kwargs
        )


    def search_ts(
        self,
        molecules,
        chg=None,
        multiplicity=None,
        method="qst",
        file_name=None,
        params=dict(),
        save_directory=None,
        check_frequency = True,
        **kwargs
    ):
        current_directory = os.getcwd()
        n = 0

        params = deepcopy(params)

        if 'opt' not in params:
            params['opt'] = []

        # Check initial hessian        
        if 'initial_hessian' in kwargs: # calcfc, readfc
            initial_hessian = kwargs['initial_hessian']
        else:
            initial_hessian = 'calcfc' # This is default ...

        if initial_hessian is not None:
            # Remove hessian related keywords ...
            params['opt'].append(initial_hessian)

        if self.nomicro:
            params['opt'].append('nomicro')

        for molecule in molecules:
            if molecule is not None:
                n += 1
        if n == 1:
            ts_molecule = molecules[-1].copy()
        else:
            ts_molecule = molecules[0].copy()
        if "opt" not in params:
            params['opt'] = []

        if "qst" in method:
            params['opt'] = params['opt'] + [f'qst{n}', 'maxcycles=1000']
            if file_name is None:
                file_name = f"qst{n}"
        elif method == "ts":
            params['opt'] = params['opt'] + ['ts', 'noeigentest','maxcycles=1000']
            if file_name is None:
                file_name = "ts"
        else:
            print("Unknown TS searching method was used !!!")
            return None, {}
        # print (ts_molecule)
        data = self.run(molecules,chg,multiplicity,file_name,params,save_directory,**kwargs)
        # Read output
        try:
            data.scfenergies
            data.atomcoords
        except:
            os.chdir(current_directory)
            return None, {}
        if len(data.atomcoords) == 0:
            os.chdir(current_directory)
            return None, {}
        converter = 1
        if self.energy_unit == 'kcal':
            converter = 23.06
        if self.energy_unit == 'Hartree':
            converter = 0.036749326681
            #converter = 1/27.2114
        index = -1
        energy = data.scfenergies[index] * converter
        coordinate_list = data.atomcoords[-1] # Last geometry corresponds to TS 
        process.locate_molecule(ts_molecule,coordinate_list,False)
        ts_molecule.energy = energy
        if not check_frequency or not data.metadata['success']:
            return ts_molecule,{} 
        
        self.move_file(file_name, save_directory) # TS file

        # TODO: This part should be done more precisely ...
        if 'freq' not in params:
            # Calculate frequency ...
            new_params=deepcopy(params)
            del(new_params['opt'])
            new_params['freq'] = [] 
            data = self.run([ts_molecule],chg,multiplicity,'freq',new_params,save_directory,**kwargs)
            self.move_file('freq', save_directory) # Freq file ...

        imaginary_vibrations = dict()
        try:
            vibmodes = data.vibdisps
            vibfreqs = data.vibfreqs
        except:
            print("TS Calculation not well converged ...")
            os.chdir(current_directory)
            return ts_molecule, {}
        m = len(vibfreqs)
        for i in range(m):
            freq = vibfreqs[i]
            mode = vibmodes[i]
            freq = round(freq, 4)
            if freq < 0:
                imaginary_vibrations[freq] = mode

        # os.system('mv new.chk old.chk')
        os.chdir(current_directory)

        return ts_molecule, imaginary_vibrations

    def run_irc(
        self,
        ts_molecule,
        chg=None,
        multiplicity=None,
        file_name="irc",
        params=dict(),
        save_directory=None,
        **kwargs
    ):
        original_content = self.content
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        
        params = deepcopy(params)

        # TODO: May be need to be handled in detail later ...
        if 'chk_file' in kwargs:
            params['irc'] = ['LQA','recorrect=never','readfc','maxpoints=60','stepsize=15','maxcycles=100']
        else:
            params['irc'] = ['LQA','recorrect=never','calcfc','maxpoints=60','stepsize=15','maxcycles=100']

        if self.nomicro:
            params['irc'].append('nomicro')

        self.make_input(
            [None, None, ts_molecule],
            chg,
            multiplicity,
            file_name,
            params,
            **kwargs
        )
        self.content = original_content
        os.system(f"{self.command} {file_name}.com")

        # Fix output
        cclib_rot_const_fix(f"{file_name}.log")
        # Read output
        p = cclib.parser.Gaussian(f"{file_name}.log")
        data = p.parse()
        os.system(f"cp {file_name}.log {file_name}_temp.log")

        irc_trajectory = []
        try:
            geometries = data.atomcoords
            energies = data.scfenergies
        except:
            print("IRC Calculation did not start properly !!!")
            os.chdir(current_directory)
            return irc_trajectory

        # split log into two
        # forward and reverse
        forward_lines = list()
        reverse_lines = list()
        with open(f"{file_name}_temp.log", 'r') as f_read:
            is_forward = True
            for line in f_read.readlines():
                if "Beginning calculation of the REVERSE path." == line.strip(): 
                    is_forward = False
                if is_forward:
                    forward_lines.append(line)
                else:
                    reverse_lines.append(line)
        assert "Calculation of FORWARD path complete." == forward_lines[-1].strip()

        with open(f"{file_name}_forward.log", 'w') as f_write:
            f_write.writelines(forward_lines)
        with open(f"{file_name}_reverse.log", 'w') as f_write:
            f_write.writelines(reverse_lines)

        # reverse
        p = cclib.parser.Gaussian(f'{file_name}_reverse.log')
        data_reverse = p.parse()

        geometries_reverse = list(reversed(data_reverse.atomcoords)) 
        energies_reverse = list(reversed(data_reverse.scfenergies)) 

        # forward
        p = cclib.parser.Gaussian(f'{file_name}_forward.log')
        data_forward = p.parse()

        geometries_forward = list(data_forward.atomcoords) 
        energies_forward = list(data_forward.scfenergies) 

        geometries = geometries_reverse + geometries_forward
        energies = np.array(energies_reverse + energies_forward)

        converter = 1
        if self.energy_unit == 'kcal':
            converter = 23.06
        if self.energy_unit == 'Hartree':
            converter = 0.036749326681
        energies *= converter

        for i,coordinate_list in enumerate(geometries):
            copied_molecule = ts_molecule.copy()
            process.locate_molecule(copied_molecule,coordinate_list,True)
            copied_molecule.energy = energies[i]
            irc_trajectory.append(copied_molecule)

        #os.system('mv new.chk old.chk')
        #print ('savesavesave',save_directory)
        #print (file_name, os.getcwd(),f'{file_name}.log')

        self.move_file(file_name, save_directory)
        os.chdir(current_directory)        
        return irc_trajectory

    def clean_scratch(self, file_name="test.com"):
        working_directory = self.working_directory
        chk_directory = os.path.join(working_directory, "old.chk")
        os.system(f"rm {chk_directory}")
        chk_directory = os.path.join(working_directory, "new.chk")
        os.system(f"rm {chk_directory}")
        # file_directory = os.path.join(working_directory,'test.com')
        # os.system(f'rm {file_directory}')
        # file_directory = os.path.join(working_directory,'test.log')
        # os.system(f'rm {file_directory}')



