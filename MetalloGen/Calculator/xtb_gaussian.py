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


class XTB_Gaussian: # Only can be combined with Gaussian! 

    #opt(nomicro,modredundant) external='/home/lkh/.conda/envs/xtb_gaussianatom/bin/python3 /home/lkh/.conda/envs/xtb_gaussianatom/lib/python3.9/site-packages/xtb_gaussianatom/geomopt.py'
    def __init__(self,command='g16',working_directory=None):
        # Check command
        check = distutils.spawn.find_executable(command)
        if check is not None:
            self.command = command
        else:
            # Check default command
            commands = ['g09','g16']
            found = False
            for old_command in commands:
                check = distutils.spawn.find_executable(old_command)
                if check is not None:
                    self.command = old_command
                    print (f'command {self.command} is used for running Gaussian, instead of {command}!')
                    found = True
                    break
            if not found:
                print ('Gaussian not found!')
                exit()

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
        # Check save directory
        xtb_directory = '\'' + os.environ['xtbbin']+'\' '
        if xtb_directory == '':
            print ('No external xtb hessian has found !!!')
            exit()
        self.content=f'#p external={xtb_directory}'
        self.energy_unit = 'Hartree'
        #self.functional = functional

    def __str__(self):
        content = ''
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
        #os.environ['GAUSS_SCRDIR'] = working_directory

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

    def write_molecule_info(self,f,molecule,chg,multiplicity):
        f.write(f'{int(chg)} {int(multiplicity)}\n')
        for atom in molecule.atom_list:
            f.write(atom.get_content())
        f.write('\n')
           

    '''def make_taskargs(self,task):
        f = open('taskargs','w')
        if self.functional == 'AIQM1':
            f.write(f'{self.functional}\n')
        else:
            functional = str.lower(self.functional)
            f.write(f'usextb_gaussianmodel\nxtb_gaussianmodelType={functional}\n')            
        f.write('xyzfile=xyz_temp.dat\n')
        f.write('yestfile=enest.dat\n')
        f.write('ygradxyzestfile=gradest.dat\n')
        if task == 'freq':
            f.write('hessianestfile=hessest.dat\n')
        f.close()'''

    def make_input(self,molecules,chg=None,multiplicity=None,file_name='test',constraints={},extra=''):
        f = open(f'{file_name}.com','w')
        if chg is None or multiplicity is None:
            if molecules[0] is not None:
                chg, multiplicity = self.get_default_mol_params(molecules[0])
            elif molecules[-1] is not None:
                chg, multiplicity = self.get_default_mol_params(molecules[-1])
        content = self.get_content()
        if len(constraints) > 0:
            if 'modredundant' not in extra:
                print ('WARNING! Fixed optimization is not working ...')
                print ('constraints:',constraints, extra)
        content = content + f'{extra}\n\n'
        f.write(content)
        labels = ['R','P','TS']
        for i, molecule in enumerate(molecules):
            if molecule is not None:
                f.write(f'{labels[i]}\n\n')
                self.write_molecule_info(f,molecule,chg,multiplicity)
        ### Write constraints
        for constraint in constraints:
            constraint_info = [] 
            if len(constraint) == 1:
                constraint_info.append('X')
            elif len(constraint) == 2:
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
        f.write('\n')
        f.close()

    def move_file(self,file_name,save_directory):
        if file_name is not None and save_directory is not None:
            current_directory = os.getcwd()
            try:
                os.chdir(self.working_directory)
            except:
                os.chdir(current_directory)
            os.system(f'mv {file_name}.com {save_directory}/')
            os.system(f'mv {file_name}.log {save_directory}/')
            os.system(f'mv *chk {save_directory}/') 

    def run(self,molecule,chg=None,multiplicity=None,file_name='test',save_directory=None): # Running with user defined result and return raw data
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        self.make_input(molecule,chg,multiplicity,file_name,{},'')
        os.system(f'{self.command} {file_name}.com')
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
        self.move_file(file_name,save_directory)
        converter = 1
        if self.energy_unit == 'kcal':
            converter = 23.06
        if self.energy_unit == 'Hartree':
            converter = 0.036749326681
        if 'opt' in self.content: 
            print ('Running optimization !!!')
            #index = np.argmin(data.scfenergies)
            #if index > len(data.atomcoords) - 1:
            #    index = -1
            index = -1
            coordinate_list = data.atomcoords[index] # Sometimes length is not matched, it performs extra scfenergy calculation
            energy = data.scfenergies[index] * converter
            process.locate_molecule(molecule,coordinate_list,False)
            molecule.energy = energy
        os.chdir(current_directory) 
    

    def get_energy(self,molecule,chg=None,multiplicity=None,file_name='sp',extra='',save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        if 'sp' not in extra:
            extra = ' sp ' + extra
        self.make_input([molecule],chg,multiplicity,file_name=file_name,extra=extra)
        #self.make_taskargs('sp')
        os.system(f'{self.command} {file_name}.com')
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
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

    def get_force(self,molecule,chg=None,multiplicity=None,file_name='force',extra=' Symmetry = None',save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        if 'force' not in extra:
            extra = ' force ' + extra
        self.make_input([molecule],chg,multiplicity,file_name=file_name,extra=extra)
        #self.make_taskargs('force')
        os.system(f'{self.command} {file_name}.com')
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
        bohr_to_angstrom = 0.529177
        self.move_file(file_name,save_directory)
        #os.system('mv new.chk old.chk')
        os.chdir(current_directory)
        return data.grads[-1]/bohr_to_angstrom

    def get_hessian(self,molecule,chg=None,multiplicity=None,file_name='hessian',extra=' Symmetry = None',save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        original_content = self.content
        if 'chk' not in self.content:
            self.content = '%chk=hessian.chk\n' + self.content
        if 'freq' not in extra:
            extra = ' freq ' + extra
        if 'Symmetry' not in extra:
            extra = extra + ' Symmetry = None '
        #self.make_taskargs('freq')
        self.make_input([molecule],chg,multiplicity,file_name=file_name,extra=extra)
        os.system(f'{self.command} {file_name}.com')
        os.system('formchk hessian.chk')
        force = []
        hessian = []
        bohr_to_angstrom = 0.529177 # Units are Hartree/bohr in chkpoint file

        # Read fchk file
        with open('hessian.fchk','r') as f:
            lines = f.readlines()
            index = 0
            while index < len(lines):
                line = lines[index]
                if 'Cartesian Gradient' in line:
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
        #os.system('mv new.chk old.chk')
        os.chdir(current_directory)
        self.content = original_content
        return force/bohr_to_angstrom,hessian/bohr_to_angstrom**2

    def optimize_geometry(self,molecule,constraints={},chg=None,multiplicity=None,file_name='test',extra='',save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        if 'opt' not in extra:
            extra = ' opt(nomicro) ' + extra
        #self.make_taskargs('opt')
        self.make_input([molecule],chg,multiplicity,file_name,constraints,extra)
        os.system(f'{self.command} {file_name}.com')
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
        self.move_file(file_name,save_directory)
        try:
            data.scfenergies
            data.atomcoords
        except:
            os.chdir(current_directory)
            return
        if len(data.atomcoords) == 0:
            os.chdir(current_directory)
            return
        converter = 1
        if self.energy_unit == 'kcal':
            converter = 23.06
        if self.energy_unit == 'Hartree':
            converter = 0.036749326681
            #converter = 1/27.2114
        energy = data.scfenergies[-1] * converter
        coordinate_list = data.atomcoords[-1]
        process.locate_molecule(molecule,coordinate_list,False)
        molecule.energy = energy
        #os.system('mv new.chk old.chk')
        os.chdir(current_directory)
        
    def relax_geometry(self,molecule,constraints,chg=None,multiplicity=None,file_name='test',num_relaxation=5,maximal_displacement=1000,save_directory=None):
        if maximal_displacement < 100:
            max_step = int(maximal_displacement*100) + 1
            if len(constraints) > 0:
                extra = f' opt(nomicro,modredundant,maxcycles={num_relaxation},maxstep={max_step},notrust) Symmetry=None'
            else:
                extra = f' opt(nomicro,maxcycles={num_relaxation},maxstep={max_step},notrust) Symmetry=None'
        else:
            if len(constraints) > 0:
                extra = f' opt(nomicro,modredundant,maxcycles={num_relaxation}) Symmetry = None'
            else:
                extra = f' opt(nomicro,maxcycles={num_relaxation}) Symmetry = None'
        self.optimize_geometry(molecule,constraints,chg,multiplicity,file_name,extra,save_directory)

    def relax_geometry_steep(
        self,
        molecule,
        constraints,
        chg=None,
        multiplicity=None,
        file_name="test",
        num_relaxation=5,
        maximal_displacement=1000,
        save_directory=None,
    ):
        if maximal_displacement < 100:
            max_step = int(maximal_displacement * 100) + 1
            if len(constraints) > 0:
                extra = f" opt(nomicro,modredundant,steep,maxcycles={num_relaxation},maxstep={max_step},notrust) Symmetry=None"
            else:
                extra = f" opt(nomicro,steep,maxcycles={num_relaxation},maxstep={max_step},notrust) Symmetry=None"
        else:
            if len(constraints) > 0:
                extra = f" opt(nomicro,modredundant,steep,maxcycles={num_relaxation}) Symmetry = None"
            else:
                extra = f" opt(nomicro,steep,maxcycles={num_relaxation}) Symmetry = None"
        self.optimize_geometry(
            molecule, constraints, chg, multiplicity, file_name, extra, save_directory
        )


    def search_ts(self,molecules,chg=None,multiplicity=None,method='qst',file_name=None,extra = '',save_directory=None,check_frequency = True):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        n = 0
        index = None
        for i,molecule in enumerate(molecules):
            if molecule is not None:
                n += 1
                if index is None:
                    index = i            
        if index is None:
            print ('Give us proper molecular geometries !!!')
            print (molecules)
            return None, {}
        ts_molecule = molecules[index].copy()
        if 'opt' not in extra:
            if 'qst' in method:
                extra = f' opt(nomicro,qst{n},maxcycles=1000) '
                if file_name is None:
                    file_name = f'qst{n}'
            elif method == 'ts':
                extra = f' opt(nomicro,ts,noeigentest,calcfc,maxcycles=1000) '
                if file_name is None:
                    file_name = 'ts'
            else:
                print ('Unknown TS searching method was used !!!')
                return None, {}
        #print (ts_molecule)
        self.make_input(molecules,chg,multiplicity,file_name=file_name,extra=extra)
        #self.make_taskargs('opt')
        os.system(f'{self.command} {file_name}.com')
        
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
        self.move_file(file_name,save_directory)
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
        energy = data.scfenergies[-1] * converter
        coordinate_list = data.atomcoords[-1] # Last geometry corresponds to TS 
        process.locate_molecule(ts_molecule,coordinate_list,False)
        ts_molecule.energy = energy
        
        #return ts_molecule #contemporary return statement. Currently, xtb cannot calculate frequency for TS.
        if not check_frequency:
            return ts_molecule, {}
        # Recalculate frequency ...
        #self.make_taskargs('freq')
        self.make_input([ts_molecule],chg,multiplicity,file_name='freq',extra=' freq') 
        os.system(f'{self.command} freq.com')
        
        # Read output
        p = cclib.parser.Gaussian(f'freq.log')
        data = p.parse()
        self.move_file('freq',save_directory)

        # Get frequency ...
        imaginary_vibrations = dict()
        try:
            vibmodes = data.vibdisps
            vibfreqs = data.vibfreqs
        except:
            print ('TS Calculation not well converged ...')
            os.chdir(current_directory)
            return ts_molecule, {}
        m = len(vibfreqs)
        for i in range(m):
            freq = vibfreqs[i]
            mode = vibmodes[i]
            freq = round(freq,4)
            if freq < 0:
                imaginary_vibrations[freq] = mode

        #os.system('mv new.chk old.chk')
        os.chdir(current_directory)
        
        return ts_molecule,imaginary_vibrations

    def run_irc(self,ts_molecule,chg=None,multiplicity=None,file_name='irc',extra = '',save_directory=None,chkpoint_file='',params = dict()):
        original_content = self.content
        if chkpoint_file != '':
            self.content = original_content + f'%chk={chkpoint_file}\n'
        current_directory = os.getcwd()
        os.chdir(self.working_directory)

        default_params = {'maxpoints':100, 'stepsize':15,'maxcycles':100}

        for key in params:
            if key in default_params:
                default_params[key] = params[key]       

        if chkpoint_file == '':
            if 'irc' not in extra:
                extra = extra + f' irc(LQA, recorrect=never, CalcFC, maxpoints={default_params["maxpoints"]}, StepSize={default_params["stepsize"]}, maxcycles={default_params["maxcycles"]}) '
        else:
            if 'irc' not in extra:
                extra = extra + f' irc(LQA, recorrect=never, maxpoints={default_params["maxpoints"]}, StepSize={default_params["stepsize"]}, maxcycles={default_params["maxcycles"]}) '
        
        self.make_input([None,None,ts_molecule],chg,multiplicity,file_name=file_name,extra=extra)
        self.content = original_content
        os.system(f'{self.command} {file_name}.com')
        
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
        os.system(f"cp {file_name}.log {file_name}_temp.log")

        self.move_file(file_name,save_directory)

        irc_trajectory = []
        try:
            geometries = data.atomcoords
            energies = data.scfenergies
        except:
            print ('IRC Calculation did not start properly !!!')
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
        os.chdir(current_directory)        
        return irc_trajectory


    def clean_scratch(self,file_name='test.com'):
        working_directory = self.working_directory
        chk_directory = os.path.join(working_directory,'old.chk')
        os.system(f'rm {chk_directory}')
        chk_directory = os.path.join(working_directory,'new.chk')
        os.system(f'rm {chk_directory}')
        #file_directory = os.path.join(working_directory,'test.com')
        #os.system(f'rm {file_directory}')
        #file_directory = os.path.join(working_directory,'test.log')
        #os.system(f'rm {file_directory}')


