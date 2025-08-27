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
from acerxn import chem
from acerxn import process


class MLPotential: # Only can be combined with Gaussian! 

    #opt(nomicro,modredundant) external='/home/lkh/.conda/envs/mlatom/bin/python3 /home/lkh/.conda/envs/mlatom/lib/python3.9/site-packages/MLatom/geomopt.py'
    def __init__(self,command='g16',functional='AIQM1',working_directory=None):
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
        ml_directory = '\''+os.environ['mlatombin']+'\''
        if ml_directory == '':
            print ('No external ml potential has found !!!')
            exit()
        self.content=f'#p external={ml_directory}'
        self.energy_unit = 'Hartree'
        self.functional = functional

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
           

    def make_taskargs(self,task,chg,multiplicity,extra=''):
        f = open('taskargs','w')
        if self.functional == 'AIQM1':
            f.write(f'{self.functional}\n')
            # Make mndokw
            imult = int((multiplicity-1)/2)

            with open('mndokw','w') as mndo:
                jop = -2
                if task == 'freq':
                    jop = 2
                mndo.write(f'iop=-22 immdp=-1 +\n')
                mndo.write(f'igeom=1 iform=1 +\n')
                mndo.write(f'jop={jop} nsav15=3 +\n')
                mndo.write(f'kharge={chg} imult={imult}')
                if 'epsi' in extra:
                    index = extra.find('epsi')
                    epsi = extra[index+5:].split()[0]
                    mndo.write(f' icosmo=3 epsi={epsi}\n')
                else: 
                    mndo.write('\n')
                
        else:
            functional = str.lower(self.functional)
            f.write(f'useMLmodel\nMLmodelType={functional}\n')            
        f.write('xyzfile=xyz_temp.dat\n')
        f.write('yestfile=enest.dat\n')
        f.write('ygradxyzestfile=gradest.dat\n')
        if task == 'freq':
            f.write('hessianestfile=hessest.dat\n')
        f.close()
        


    def make_input(self,molecules,chg=None,multiplicity=None,task='sp',file_name='test',constraints={},extra=''):
        f = open(f'{file_name}.com','w')
        if chg is None or multiplicity is None:
            if molecules[0] is not None:
                chg, multiplicity = self.get_default_mol_params(molecules[0])
            elif molecules[-1] is not None:
                chg, multiplicity = self.get_default_mol_params(molecules[-1])
        content = self.get_content()
        self.make_taskargs(task,chg,multiplicity,extra)
        if len(constraints) > 0:
            if 'modredundant' not in extra:
                print ('WARNING! Fixed optimization is not working ...')
                print ('constraints:',constraints, extra)
        if 'epsi' in extra:
            index = extra.find('epsi')
            new_extra = extra[:index]
            content = content + f'{new_extra}\n\n'
        else:
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
            file_directory = f'{file_name}.com'
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
            index = np.argmin(data.scfenergies)
            if index > len(data.atomcoords) - 1:
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
        chg,multiplicity = self.make_input([molecule],chg,multiplicity,task='sp',file_name=file_name,extra=extra)
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
        self.make_input([molecule],chg,multiplicity,task='force',file_name=file_name,extra=extra)
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
        self.make_input([molecule],chg,multiplicity,task='freq',file_name=file_name,extra=extra)
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
        self.make_input([molecule],chg,multiplicity,'opt',file_name,constraints,extra)
        os.system(f'{self.command} {file_name}.com')
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
        self.move_file(file_name,save_directory)
        print(data.scfenergies)

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
        
    def relax_geometry(self,molecule,constraints,chg=None,multiplicity=None,file_name='test',num_relaxation=5,maximal_displacement=1000,save_directory=None):
        if maximal_displacement < 100:
            max_step = int(maximal_displacement*100) + 1
            if len(constraints) > 0:
                extra = f' opt(nomicro,modredundant,loose,maxcycles={num_relaxation},maxstep={max_step},notrust) Symmetry=None'
            else:
                extra = f' opt(nomicro,loose,maxcycles={num_relaxation},maxstep={max_step},notrust) Symmetry=None'
        else:
            if len(constraints) > 0:
                extra = f' opt(nomicro,modredundant,loose,maxcycles=15) Symmetry = None'
            else:
                extra = f' opt(nomicro,loose,maxcycles=15) Symmetry = None'
        self.optimize_geometry(molecule,constraints,chg,multiplicity,file_name,extra,save_directory)


    def search_ts(self,molecules,chg=None,multiplicity=None,method='qst',file_name=None,extra = '',save_directory=None):
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
        chg, multiplicity = self.make_input(molecules,chg,multiplicity,task='opt',file_name=file_name,extra=extra)
        os.system(f'{self.command} {file_name}.com')
        
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
        self.move_file(file_name,save_directory)
        # Get minimal energy geometry
        try:
            index = np.argmin(data.scfenergies)
        except:
            print ('TS Calculation did not start properly ...')
            os.chdir(current_directory)
            return None, {}
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
        process.locate_molecule(ts_molecule,coordinate_list,False)
        ts_molecule.energy = energy
       
        # Recalculate frequency ...
        self.make_input(ts_molecule,chg,multiplicity,task='freq',file_name='freq',extra=' freq') 
        os.system(f'{self.command} freq.com')
        
        # Read output
        p = cclib.parser.Gaussian(f'freq.log')
        data = p.parse()
        self.move_file(file_name,save_directory)

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

    def run_irc(self,ts_molecule,chg=None,multiplicity=None,file_name='irc',extra = '',save_directory=None,chkpoint_file=''):
        original_content = self.content
        if chkpoint_file != '':
            self.content = original_content + f'%chk={chkpoint_file}\n'
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        
        if chkpoint_file == '':
            if 'irc' not in extra:
                extra = extra + f' irc(LQA, recorrect=never, CalcFC, maxpoints=60, StepSize=15, maxcycles=100) '
        else:
            if 'irc' not in extra:
                extra = extra + f' irc(LQA, recorrect=never, maxpoints=60, StepSize=15, maxcycles=100) '
        
        self.make_input([None,None,ts_molecule],chg,multiplicity,file_name=file_name,extra=extra)
        self.content = original_content
        os.system(f'{self.command} {file_name}.com')
        
        # Read output
        p = cclib.parser.Gaussian(f'{file_name}.log')
        data = p.parse()
        self.move_file(file_name,save_directory)

        irc_trajectory = []
        try:
            geometries = data.atomcoords
            energies = data.scfenergies
        except:
            print ('IRC Calculation did not start properly !!!')
            os.chdir(current_directory)
            return irc_trajectory

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


