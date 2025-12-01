'''
--chem.py--
Declaring classes.
(Intermediates, molecules, atoms, ...)
Some parts can be replaced by ASE formats, but now we are using these settings for convenient customization.
'''

from scipy import spatial
import numpy as np
import itertools

from rdkit.Chem import AllChem
from rdkit import Chem

from MetalloGen.utils import make_smiles
from MetalloGen.utils import ic
from MetalloGen import process

class Atom:
    """
    :class Atom:
        class Atom mainly contains atomic_number, element, and x,
        Other attributes are not widely used
        molecule_index shows on which molecule that this atom is contained in. For example, if we consider Intermediate C.C, 
        molecule_index can have either 0 or 1, and every atom within atom_list of Intermediate can be assigned by checking
        which molecule does the atom belong to.

    :param data(str or integer):
        Data is either symbol that represents element or atomic number
    
    """
    global periodic_table
    periodic_table = ['H','He','Li','Be','B','C','N','O','F','Ne',\
    'Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn',\
    'Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',\
    'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',\
    'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',\
    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',\
    'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm']
    
    global radius_dict
    radius_dict = dict( # reference: Dalton Trans., 2008, 2832-2838
    H  = 0.31, He = 0.28, \
    Li = 1.28, Be = 0.96, B  = 0.84, C  = 0.76, N  = 0.71, O  = 0.66, F  = 0.57, Ne = 0.58, \
    Na = 1.66, Mg = 1.41, Al = 1.21, Si = 1.11, P  = 1.07, S  = 1.05, Cl = 1.02, Ar = 0.76, \
    K  = 2.03, Ca = 1.76, Sc = 1.70, Ti = 1.60, V  = 1.53, Cr = 1.39, Mn = 1.50, Fe = 1.42, Co = 1.38, Ni = 1.24, Cu = 1.32, Zn = 1.22, Ga = 1.22, Ge = 1.20, As = 1.19, Se = 1.20, Br = 1.20, Kr = 1.16, \
    Rb = 2.20, Sr = 1.95, Y  = 1.90, Zr = 1.75, Nb = 1.64, Mo = 1.54, Tc = 1.47, Ru = 1.46, Rh = 1.42, Pd = 1.39, Ag = 1.45, Cd = 1.44, In = 1.42, Sn = 1.39, Sb = 1.39, Te = 1.38, I  = 1.39, Xe = 1.40, \
    Cs = 2.44, Ba = 2.15, La = 2.07, Ce = 2.04, Pr = 2.03, Nd = 2.01, Pm = 1.99, Sm = 1.98, Eu = 1.98, Gd = 1.96, Tb = 1.94, Dy = 1.92, Ho = 1.92, Er = 1.89, Tm = 1.90, Yb = 1.87, Lu = 1.87, Hf = 1.75, Ta = 1.70, W = 1.62, Re = 1.51, Os = 1.44, Ir = 1.41, Pt = 1.36, Au = 1.36, Hg = 1.32, Tl = 1.45, Pb = 1.46, Bi = 1.48, Po = 1.40, At = 1.50, Rn = 1.50, \
    Fr = 2.60, Ra = 2.21, Ac = 2.15, Th = 2.06, Pa = 2.00, U  = 1.96, Np = 1.90, Pu = 1.87, Am = 1.80, Cm = 1.69, D  = 0.31
    )

    def __init__(self,data = None):
        self.atomic_number = None
        self.element = None
        self.x = 0.00
        self.y = 0.00
        self.z = 0.00
        self.molecule_index = 0
        self.is_divalent_hydrogen = False
        if data is not None:
            if type(data) == str:
                self.element = data
            else:
                self.atomic_number = data

    def set_atomic_number(self,atomic_number):
        self.atomic_number = atomic_number
        self.element = periodic_table[atomic_number-1]

    def set_element(self, element):
        """ Type of an atom. e.g. 'C', 'H', 'O', and so on.""" 
        self.element = element
        self.atomic_number = periodic_table.index(element) + 1

    def set_x(self, x):
        """ X-coordinate """ 
        self.x = x

    def set_y(self, y):
        """ Y-coordinate """ 
        self.y = y

    def set_z(self, z):
        """ Z-coordinate """ 
        self.z = z

    def set_coordinate(self, position):
        """ Set Cartesian coordinates of an atom """ 
        dim = len(position)
        if dim == 2:
            x = position[0]
            y = position[1]
            z = 0
        elif dim == 3:
            x = position[0]
            y = position[1]
            z = position[2]
        self.x = x
        self.y = y
        self.z = z

    def set_molecule_index(self,index):
        self.molecule_index = index

    def set_is_divalent_hydrogen(self,divalency):
        self.is_divalent_hydrogen = divalency

    def set_atom_type(self, atom_type):
        """ An attribute for defining atom types used in molecular mechanics calculations (e.g. C-CT, H-HC, and so on.) """ 
        self.atom_type = atom_type

    def set_configuration(self,configuration):
        self.configuration = configuration

    def set_is_active(self, is_active):
        """ True or False. Whether an atom is assigned as active one or not""" 
        self.is_active = is_active

    def get_atomic_number(self):
        """
        Returns the atomic number (number of protons) of a given atom.

        :param:

        :return integer:
            Directly the atomic number of a given atom is returned
        """
        if self.atomic_number == None:
            element = self.element
            if element == None:
                print ('atom is not specified!')
            if len(element)>1:
                end_part = element[1:]
                end_part = str.lower(end_part)
                element = element[0]+end_part
                self.element = element
            if element in periodic_table:
                index = periodic_table.index(element)
                self.atomic_number = index + 1
            elif element == 'D':
                return 1
            else:
                print ('element',element)
                print ('modify periodic table!!!')
        return self.atomic_number

    def get_element(self):
        """
        Returns symbol of a given atom.

        :param:

        :return str:
            Directly the symbol of a given atom is returned
        """
        if self.element is None:
            atomic_number = self.atomic_number
            if atomic_number is None:
                print ('atom is not specified!')
            z = int(self.atomic_number)-1
            self.element = periodic_table[z]
        return self.element

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def get_coordinate(self):
        return np.array([self.x,self.y,self.z])    

    def get_molecule_index(self):
        return self.molecule_index

    def get_is_divalent_hydrogen(self):
        return self.is_divalent_hydrogen

    def get_atom_type(self):
        try:
            return self.atom_type
        except:
            print ('Set atom type!!!')
            return None

    def get_configuration(self):
        try:
            return self.configuration
        except:
            print ('Set configuration!!!')
            return None

    def get_is_active(self):
        try:
            return self.is_active
        except:
            print ('Set activity!!!')
            return None

    # Process element information in advance
    def get_electron_count(self):
        element_type = self.get_element()
        num_electrons = dict(   SC=3,TI=4,V=5,CR=6,MN=7,FE=8,CO=9,NI=10,CU=11,ZN=12,\
                                Y=3,ZR=4,NB=5,MO=6,TC=7,RU=8,RH=9,PD=10,AG=11,CD=12,\
                                LU=3,HF=4,TA=5,W=6,RE=7,OS=8,IR=9,PT=10,AU=11,HG=12)
        return num_electrons[element_type]

    def get_electronegativity(self):
        """
        Returns the electronegativity of a given atom.

        :param:

        :return float:
            Directly the electronegativity of a given atom is returned
        """
        element = self.get_element()
        a = str.lower(element)
        if a=='n': 
            return 3.0
        elif a=='c': 
            return 2.5
        elif a=='s': 
            return 2.5
        elif a=='o': 
            return 3.5
        elif a=='f': 
            return 4.0
        elif a=='br': 
            return 2.8
        else: 
            return 0
       
    def get_mass(self):
        """
        Returns the exact value (float) of an atomic mass of a given atom.

        :param:

        :return float:
            Directly the atomic mass of a given atom is returned
        """
        element = self.get_element()
        a = str.lower(element)
        if a=='h': return 1.008
        elif a=='he': return 4.003
        elif a=='li': return 6.941
        elif a=='be': return 9.012
        elif a=='b': return 10.81
        elif a=='c': return 12.01
        elif a=='n': return 14.01
        elif a=='o': return 16.00
        elif a=='f': return 19.00
        elif a=='ne': return 20.18
        elif a=='na': return 22.99
        elif a=='mg': return 24.31
        elif a=='al': return 26.98
        elif a=='si': return 28.09
        elif a=='p': return 30.97
        elif a=='s': return 32.07
        elif a=='cl': return 35.45
        elif a=='ar': return 39.95
        elif a=='k': return 39.10
        elif a=='ca': return 40.08
        elif a=='au': return 196.97
        elif a=='co': return 58.9332
        elif a=='ni': return 58.6934
        elif a=='ti': return 47.8671
        elif a=='fe': return 55.845
        elif a=='br': return 79.904
        elif a=='rh': return 102.90550
        elif a=='pd': return 106.42
        elif a=='hf': return 178.49
        elif a=='i': return 126.90447
        else: return 0

    def get_radius(self):
        """
        Returns a radius information of a given atom. Reference is given here: Dalton Trans., 2008, 2832-2838
        
        :param:

        :return float:
            It directly returns the reference values
        """
        element = self.get_element()       
        a = str.lower(element)
        a = a.capitalize()
        return radius_dict[a]

    def get_period_group(self):
        """
        Returns a period,group information from a given atom. It finds period, group by identifying 
        electron configuration (orbital configuration)
        
        :param:

        :return period,group(int,int):
            Note that these values are actual period and group. If C atomm is given, it returns 2,4

        """
        atomic_number = self.get_atomic_number()
        num_of_electrons = atomic_number
        # Orbital: [n,l,num_of_electrons]
        sum_of_n_and_l=1
        orbital_configuration=[]
        while num_of_electrons>0:
            # Generate orbitals within sum_of_n_and_l=k
            # New orbitals are introduced for (k+1)/2
            maximal_l=int((sum_of_n_and_l-1)/2)
            for l in range(maximal_l,-1,-1):
                # Start with lowest l
                if num_of_electrons>4*l+2:
                    num_of_electrons-=(4*l+2)
                    orbital_configuration.append([sum_of_n_and_l-l,l,4*l+2])
                else:
                    orbital_configuration.append([sum_of_n_and_l-l,l,num_of_electrons])
                    num_of_electrons=0
                    break
            sum_of_n_and_l+=1
        # Get maximal n and l
        period=0
        for orbital in orbital_configuration:
            if orbital[0]>period:
                period = orbital[0]
        # If transition metal, we add 9, Sc has group 9 for else, we do not consider ...
        last_orbital = orbital_configuration[-1]
        if last_orbital[1]<2:
            group = 2 * last_orbital[1] ** 2 + last_orbital[2]
        else:
            group = 8 + last_orbital[2]
        return period,group

    def get_electron_count(self):
        """
        """
        element = self.get_element()
        num_electrons = dict(   SC=3,TI=4,V=5,CR=6,MN=7,FE=8,CO=9,NI=10,CU=11,ZN=12,\
                                Y=3,ZR=4,NB=5,MO=6,TC=7,RU=8,RH=9,PD=10,AG=11,CD=12,\
                                LU=3,HF=4,TA=5,W=6,RE=7,OS=8,IR=9,PT=10,AU=11,HG=12)
        return num_electrons[element]

    def get_max_valency(self): 
        """
        Returns maximal valency of a given atom. Examples of those values are shown in the main code.  

        :param:

        :return integer(nonnegative integer):
            possible maximal valency
        """
        element = self.get_element()
        a = str.lower(element)
        if a=='c':   
            return 4
        elif a=='h': 
            return 1
        elif a=='o': 
            return 3
        elif a=='n': 
            return 4
        elif a=='p':
            return 5 #valence shell expansion
        elif a=='s': 
            return 6 #valence shell expansion
        elif a=='f': 
            return 1
        elif a=='li': 
            return 1
        elif a=='b': 
            return 3
        elif a=='na': 
            return 1
        elif a=='mg':
            return 2
        elif a=='al':
            return 3
        elif a=='co':
            return 6
        elif a=='rh':
            return 6
        elif a=='ni':
            return 6 
        elif a=='ti':
            return 6
        elif a=='fe':
            return 6
        elif a=='cl': 
            return 1
        elif a=='br': 
            return 1
        elif a=='bb': 
            return 3
        elif a=='lg': 
            return 2
        elif a=='pd': 
            return 6
        elif a=='i': 
            return 3
        elif a=='zr':
            return 4
        elif a=='si':
            return 4
        elif a=='mn':
            return 6
        else:
            return 6

    def copy(self):
        new_atom = Atom()
        # Copy all attributes
        new_atom.atomic_number = self.atomic_number
        new_atom.element = self.element
        new_atom.x = self.x
        new_atom.y = self.y
        new_atom.z = self.z
        return new_atom 

    def is_same_atom(self,atom):
        """
        Returns whether the two atoms have same type by comparing atomic number

        :param atom(pyclass 'Atom'):
            Our defined class 'Atom'

        :return:
            True: Two atoms are same type
            False: Two atoms are different type
        """
        atomic_number = self.get_atomic_number()
        atomic_number_prime = atom.get_atomic_number()
        return atomic_number == atomic_number_prime

    def get_content(self,option='element',criteria = 1e-4):
        x = self.x
        y = self.y
        z = self.z
        if abs(x) < criteria:
            x = 0.00
        if abs(y) < criteria:
            y = 0.00
        if abs(z) < criteria:
            z = 0.00
        content = ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
        if option=='element':
            content = self.get_element() + content
        else:
            content = str(self.get_atomic_number()) + content
        return content

    def __eq__(self,atom):
        return self.is_same_atom(atom)

         
class Molecule:
    """
    :class Molecule:
        class Molecule mainly contains atom_list, atom_feature, chg, bo_matrix, adj_matrix, energy, smiles, c_eig_list
        atom_list is a list of atom (pyclass 'Atom')
        atom_feature is a dict, where features of atom such as formal charge, number of pi bonds, etc, are stored.
        Those information can be freely added by giving new dict
        c_eig_list can be used to identify whether the given two molecules are the same. This c_eig_list is invariant
        to the permutation of atom indexing, identification between any two generated molecules can be easily checked.

    :param data(str or xyz file or None):
        data should be feeded as either smiles(str), xyz file(file) or None
        * If smiles is used, rd_mol is generated using the given smiles and converted into ace_mol (pyclass 'Molecule)
        * If xyz file is used, directly the 3d geometry of the given molecule is generated. If you want to generate adj_matrix,
        bo_matrix, chg_list, etc, refer following method contained in pyclass 'Molecule' (charge of molecule should be given!!!)
        i. Generate adj_matrix by using 'get_adj_matrix_from_distance' stored in class 'Molecule'
        ii. Generate bo_matrix by using 'get_adj_matrix_from_adj_matrix' stored in class 'Molecule'
        iii. Then, using those bo_matrix, get chg_list by using 'get_chg_list_from_bo_matrix' stored n class 'Molecule'
        * If None is used, only blank virtual molecule is generated

    """
    def __init__(self,data = None):
        self.atom_list = []
        self.atom_feature = dict()
        self.adj_matrix = None
        self.bo_matrix = None
        self.chg = None
        self.multiplicity = None
        self.energy = None        
        self.center_of_mass = None
        self.smiles = None
        self.c_eig_list = None
        self.formula_id = None
        self.molecule_id = None
        self.atom_id_list = None
        
        if data == None:
            pass

        elif type(data) == str:
            if data[-4:] == '.xyz':
                # data is already opened file
                f = open(data,'r')
                atom_list=[]
                try:
                    atom_num = int(f.readline())
                except:
                    print ('Wrong format! Should start with number of atoms!')
                try:
                    energy = float(f.readline())
                    self.energy = energy
                except:
                    self.energy = None
                for i in range(atom_num):
                    try:
                        content = f.readline().strip()
                        atom_line = content.split()
                        element_symbol = atom_line[0]
                        x = float(atom_line[1]) 
                        y = float(atom_line[2]) 
                        z = float(atom_line[3])
                        new_atom = Atom(element_symbol)
                        new_atom.x = x
                        new_atom.y = y
                        new_atom.z = z
                        atom_list.append(new_atom)
                    except:
                        print ('Error found in:',content)
                        print ('Check the file again:',data)
                self.atom_list = atom_list
                # At least make adjacency
                self.adj_matrix = process.get_adj_matrix_from_distance(self)
            elif data[-4:] == '.com':
                f = open(data,'r')
                atom_list=[]
                try:
                    info = f.readline().strip().split()
                    chg = int(info[0])
                    multiplicity = int(info[1])
                    self.chg = chg
                    self.multiplicity = multiplicity
                except:
                    print ('Wrong format! Should start with number of atoms!')
                while True:
                    try:
                        content = f.readline().strip()
                        atom_line = content.split()
                        element_symbol = atom_line[0]
                        x = float(atom_line[1]) 
                        y = float(atom_line[2]) 
                        z = float(atom_line[3])
                        new_atom = Atom(element_symbol)
                        new_atom.x = x
                        new_atom.y = y
                        new_atom.z = z
                        atom_list.append(new_atom)
                    except:
                        break
                self.atom_list = atom_list
                # At least make adjacency
                self.adj_matrix = process.get_adj_matrix_from_distance(self,1.3)
            else:
                # Generate data with rdkit
                from rdkit import Chem            
                if True:
                    rd_mol = Chem.MolFromSmiles(data,sanitize = False)
                    Chem.SanitizeMol(rd_mol,sanitizeOps = Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
                    ace_mol = process.get_ace_mol_from_rd_mol(rd_mol)
                else:
                    from openbabel import pybel
                    ob_mol = pybel.readstring('smi',data)
                    ace_mol = process.get_ace_mol_from_ob_mol(ob_mol)
                self.atom_list = ace_mol.atom_list
                self.atom_feature = ace_mol.atom_feature
                self.adj_matrix = ace_mol.adj_matrix
                self.bo_matrix = ace_mol.bo_matrix
                self.chg = np.sum(self.atom_feature['chg'])
                self.smiles = data
          
        elif type(data) == dict: # At least graph information is given ... If not, use xyz information
            atom_feature = {}
            # Check z
            if 'z' not in data: 
                atom_list = None
            else:
                atom_list = [Atom(z) for z in data['z']]
            # Next, check molecule charge
            if 'chg' in data:
                chg = data['chg']
            elif 'atom chg' in data:
                atom_feature['chg'] = data['atom chg']
                chg = np.sum(data['atom chg'])
            else:
                chg = None

            adj_matrix = None
            bo_matrix = None
            
            # Next, check adj matrix (and bo matrix)
            if 'bo' in data:
                bo_matrix = data['bo']
            if 'adj' in data:
                adj_matrix = data['adj']
            else:
                if bo_matrix is not None:
                    adj_matrix = np.where(bo_matrix>0,1,0)           

            # Check energy
            if 'energy' in data:
                self.energy = data['energy']

            # If geometry exists, update the geometry to atoms
            if 'coords' in data:
                process.locate_molecule(self,data['coords'])
                # If adj is not given ...
                if adj_matrix is None:
                    adj_matrix = process.get_adj_matrix_from_distance(self,1.2)

            self.atom_list = atom_list
            self.chg = chg
            self.adj_matrix = adj_matrix
            self.bo_matrix = bo_matrix
            self.atom_feature = atom_feature

        else:
            try:
                len(data[0]) # Then, it's some kind of list,tuple,numpy
                make = True
            except:
                make = False
            if make:
                # Data: (z_list,adj_matrix,bo_matrix,chg_list)
                atom_list = []
                atom_info_list = data[0]
                for atom_info in atom_info_list:
                    atom = Atom(atom_info)
                    atom_list.append(atom)
                self.atom_list = atom_list
                self.adj_matrix = data[1]
                self.bo_matrix = data[2]
                self.atom_feature['chg'] = data[3]
                if self.adj_matrix is None and self.bo_matrix is not None:
                    self.adj_matrix = np.where(self.bo_matrix>0,1,0)
            else:
                a = 1 

    def reset_molecule(self):
        self.atom_list = None
        self.atom_feauture = dict()
        self.adj_matrix = None
        self.bo_matrix = None
        self.energy = None
        self.smiles = None
        self.chg = None
        try:
            self.molecule_feature = dict()
        except:
            b = 1
        self.c_eig_list = None
        self.formula_id = None
         
    def set_atom_list(self,atom_list):
        self.atom_list = atom_list
        self.atom_feature['atomic number'] = self.get_z_list()

    def set_atom_feature(self,atom_feature,key):
        # Caution: Only change atom features that are related to max
        self.atom_feature[key] = atom_feature
        
    def set_adj_matrix(self,adj_matrix):
        self.atom_feature['chg'] = None
        if adj_matrix is not None:
            self.adj_matrix = np.copy(adj_matrix)
        else:
            self.adj_matrix = None
        self.bo_matrix = None
        
    def set_bo_matrix(self,bo_matrix):
        self.atom_feature['chg'] = None
        self.bo_matrix = np.copy(bo_matrix)
        self.adj_matrix = np.where(self.bo_matrix>0,1,0)

    def set_chg(self,chg):
        self.chg = chg
    
    def set_multiplicity(self,multiplicity):
        chg = self.get_chg()
        if chg is None:
            print ('Cannot set multicplity since the total charge is not specified!')
        else:
            num_electron = -chg + np.sum(self.get_z_list())
            if num_electron%2 != (multiplicity + 1)%2:
                print (f'There are {num_electron} electrons, therefore spin with {multiplicity} is impossible!')
            else:
                self.multiplicity = multiplicity
        
    def set_energy(self,energy):
        self.energy = energy

    def set_smiles(self,smiles):
        self.smiles = smiles

    def set_num_rings(self,num_rings):
        self.num_rings = num_rings

    def set_c_eig_list(self,c_eig_list):
        self.c_eig_list = c_eig_list

    def set_formula_id(self,formula_id):
        self.formula_id = formula_id

    def set_molecule_id(self,molecule_id):
        self.molecule_id = molecule_id

    def set_associated_fragment_list(self,associated_fragment_list):
        self.associated_fragment_list = associated_fragment_list

    def set_original_indices(self,indices):
        """
        Only used for fragment based methods
        """
        self.original_indices = indices

    def set_center_of_mass(self,center_of_mass):
        self.center_of_mass = center_of_mass

    def set_center(self,center):
        self.center = center

    def set_screening_result(self,is_screened,screening_log):
        self.is_screened = is_screened
        self.screening_log = screening_log

    def set_atom_num(self,atom_num):
        self.atom_num = atom_num

    def set_atom_info(self,info,key):
        self.atom_feature[key] = info 

    def setr(self, r):
        self.r=r

    def sett(self, t):
        """ Spherical coordinate of the CM of a molecule. Used in Basin-hopping sampling. """ 
        self.t=t

    def setp(self, p):
        """ Spherical coordinate of the CM of a molecule. Used in Basin-hopping sampling. """ 
        self.p=p

    def setdr(self, dr):
        """ A parameter for defining the range of random movement of a molecular framgent. Used in Basin-hopping sampling. """ 
        self.dr=dr

    def set_sssr(self,sssr):
        self.sssr = sssr

    def update(self,information):
        if information == 'geometry':
            self.center = self.get_center()
            self.center_of_mass = self.get_center_of_mass()
            self.atom_feature['radius'] = self.get_radius_list()
            self.atom_feature['coords'] = self.get_coordinate_list()
        elif information == 'id':
            self.c_eig_list = self.get_c_eig_list()
        elif information == 'atom feature':
            period_list,group_list = self.get_period_group_list()
            self.atom_feature['atomic number'] = self.get_z_list()
            self.atom_feature['element'] = self.get_element_list()
            self.atom_feature['period'] = period_list
            self.atom_feature['group'] = group_list
            self.atom_feature['max valency'] = self.get_max_valency_list() 
            self.atom_feature['max bo'] = self.get_max_bo_list()

    def get_atom_list(self):
        return self.atom_list
    
    def get_atom_feature(self):
        return self.atom_feature
    
    def get_adj_matrix(self):
        if self.adj_matrix is not None:
            return self.adj_matrix
        if self.bo_matrix is not None:
            adj_matrix = np.where(self.bo_matrix>0,1,0)
            return adj_matrix
        return None
    
    def get_bo_matrix(self):            
        return self.bo_matrix
    
    def get_chg(self):
        if self.chg is None:
            chg_list = self.get_chg_list()
            if chg_list is None:
                return None
            else:
                return int(np.sum(chg_list))
        else:
            return int(self.chg)

    def get_multiplicity(self):
        if self.multiplicity is None:
            try:
                chg = self.get_chg()
            except:
                print ('Total charge is not provided! At least charge information!')
                return None
            try: # If bo and chg is present ...
                e_list = self.get_num_of_lone_pair_list()
                num_electron = len(np.where((2*e_list) % 2 == 1)[0])    
            except: # If either one is not present ...
                z_sum = np.sum(self.get_z_list())
                num_electron = z_sum - chg
            multiplicity = num_electron % 2 + 1
            return int(multiplicity)
        else:
            return int(self.multiplicity)

    def get_molecule_feature(self):
        return self.molecule_feature

    def get_energy(self):
        return self.energy

    def initialize(self,method='SumofFragments'):
        adj_matrix = self.get_adj_matrix()
        chg = self.get_chg()
        if adj_matrix is None or chg is None:
            print ('Cannot initialize bo and chgs!')
        fc_list,bo_matrix = process.get_chg_and_bo(self,chg,method=method)
        self.bo_matrix = bo_matrix
        self.atom_feature['chg'] = fc_list
        

    def get_smiles(self,method = 'SumofFragments',find_stereocenter='N'):
        """
        Returns smiles of a given molecule. It could return different smiles according to the atom order.

        :param method(str):
            method can either be 'ace' or 'rdkit'
            If 'ace' is used, smiles is generated throughout our defined method
            If 'rdkit' is used, it generates smiles using rdkit module
             
        :param find_stereocenter(str):
            It can have either 'Y' or 'N'. If 'Y'(yes), stereo information such as E/Z or R/S are desginated in the smiles
        
        :return smiles(str):
            Returns a smiles of given molecule generated by input option
        """
        # If bo and chg is both present, use the value. If not, simply recalculate bo and chg ...
        atom_list = self.atom_list
        bo_matrix = self.get_matrix('bo')
        fc_list = self.get_chg_list()
        if fc_list is None or bo_matrix is None:
            chg = self.get_chg()
            adj_matrix = self.get_adj_matrix()
            if chg is None or adj_matrix is None:
                print ('We need to know both adjacency and charge!!!')
                return None
            fc_list, bo_matrix = process.get_chg_and_bo(self,chg=chg,method=method)
        fc_list = fc_list.tolist()
        # Check element is ready!
        for atom in atom_list:
            atom.set_element(atom.get_element())
        bond_list = self.get_bond_list(False)
        smiles_list = make_smiles.GetSMILES(atom_list,bo_matrix,bond_list,fc_list,find_stereocenter)
        return smiles_list[0]

    def get_smiles_with_rdkit(self,include_hydrogen = False):
        rd_mol = self.get_rd_mol()
        if not include_hydrogen:
            rd_mol = Chem.rdmolops.RemoveHs(rd_mol)
        return Chem.MolToSmiles(rd_mol)
        
    def get_c_eig_list(self,c_sum = False):
        """
        Returns the eigenvalues of coulomb matrix (see the instruction 'get_matrix' within class 'Molecule'.

        :param c_sum(boolean):
            If True, it also returns the sum of eigenvalues

        :return c_eig_list(pyclass 'numpy.ndarray' or pyclass 'numpy.ndarray',int(c_matrix)):
            eigenvalue of coulomb matrix of a given molecule and also it's sum if c_sum = True

        """
        if self.c_eig_list is None:        
            c_matrix = self.get_matrix('coulomb')
            #### Check symmetry of matrix
            check_sum = np.sum(np.abs(c_matrix-np.transpose(c_matrix)))
            if check_sum > 0.01:
                print ('something wrong')
            c_eig_list,vec = np.linalg.eig(c_matrix)
            if np.isnan(c_eig_list[0]):
                print ('nanvalue!!!')
                import sys;sys.exit()
            if c_sum:
                return np.sort(c_eig_list.real), int(np.sum(c_matrix))
            else:
                return np.sort(c_eig_list.real)
        else:
            return self.c_eig_list

    def get_formula_id(self):
        if self.formula_id is not None:
            return self.formula_id
        formula_id = int(np.sum(self.get_z_list()**3))
        return formula_id

    def get_atom_id_list(self):
        if 'id' in self.atom_feature:
            return self.atom_feature['id']
        distance_matrix = self.get_distance_matrix('graph')
        n = len(self.atom_list)
        z_list = self.get_z_list()
        z_matrix = np.repeat(z_list,n)
        z_matrix = z_matrix.reshape((n,n))
        score_matrix = (z_matrix ** 3) * (distance_matrix + 1) ** 2
        atom_id_list = np.sum(score_matrix,axis=0)
        return atom_id_list

    def get_connectivity_id(self):
        atom_id_list = self.get_atom_id_list()
        return int(np.sum(atom_id_list))

    def get_molecule_id(self):
        if self.molecule_id is not None:
            return self.molecule_id
        chg = int(self.get_chg())
        connectivity_id = self.get_connectivity_id()
        return (chg,connectivity_id)

    def get_associated_fragment_list(self):
        try:
            return self.associated_fragment_list
        except:
            print ('Set fragment list!!!')
            return None

    def get_center_of_mass(self):
        try:
            return self.center_of_mass
        except:
            atom_list = self.atom_list
            coordinate_list = self.get_coordinate_list()
            mass_list = self.get_mass_list()
            total_mass = np.sum(mass_list)
            mass_list = np.expand_dims(mass_list,-1)
            weighted_coords = coordinate_list * mass_list
            center_of_mass = np.sum(weighted_coords,axis=0)/total_mass
            return center_of_mass

    def get_center(self):
        try:
            return self.center
        except:
            atom_list = self.atom_list
            coordinate_list = self.get_coordinate_list()
            coordinate_list = np.array(coordinate_list)
            center = np.mean(coordinate_list,axis=0)
            return center 
    
    def get_screening_result():
        try:
            return self.is_screened,self.screening_log
        except:
            print ('Set screening result!!!')
            return None

    def getr(self):
        try:
            return self.r
        except:
            print ('Set r!!!')
            return None
    
    def gett(self):
        try:
            return self.t
        except:
            print ('Set t!!!')
            return None

    def getp(self):
        try:
            return self.p
        except:
            print ('Set p!!!')
            return None

    def getdr(self):
        try:
            return self.dr
        except:
            print ('Set dr!!!')
            return None
            
    def get_matrix(self,type_of_matrix = 'bo'):
        """
        Returns a matrix that contains some information of a given molecule that is widely used. 
        Distance matrix can be easily evaluated by using rdkit module. 
        :param type_of_matrix(str):
            Mainly 4 inputs are possible: 'bo', 'adj', 'coulomb', 'distance'
            'bo' corresponds to bond order matrix (B_{ij})
            'adj' corresponds to adjacency matrix whose elements (A_{ij}) are 1 for bonded i,jth atom and 0 for nonbonded i,jth atom 
            'coulomb' corresponds to the coulomb matrix whose elements are represented as C_{ij} = A_{ij}*Z_{i}*Z_{j}
            'distance' corresponds to the graphical distance matrix (not actual distance matrix). 
            D_{ij} is the length of shortest path between ith atom and jth atom

        :return matrix(pyclass numpy.ndarray):
         
        """
        atom_list = self.atom_list
        if type_of_matrix == 'bo':
            return self.get_bo_matrix()
        elif type_of_matrix == 'adj':
            return self.get_adj_matrix()
        elif type_of_matrix == 'coulomb':
            adj_matrix = self.get_adj_matrix()
            z_list = self.get_z_list()
            if z_list is None:
                return None
            new_adj_matrix = adj_matrix + np.diag([1]*len(z_list))
            diagonal_matrix = np.diag(z_list)
            coulomb_matrix = np.matmul(np.matmul(diagonal_matrix,new_adj_matrix),diagonal_matrix)
            return coulomb_matrix
        elif type_of_matrix == 'distance':
            adj_matrix = self.get_adj_matrix()
            n = len(self.atom_list)
            distance_matrix = 1000 * np.ones((n,n))
            np.fill_diagonal(distance_matrix,0)
            update_matrix = np.identity(n)
            for d in range(n):
                cur_d = d + 1
                update_matrix = np.matmul(update_matrix,adj_matrix)
                d_update_matrix = cur_d * np.where(update_matrix>0,1,1000)
                indices = np.where(d_update_matrix<distance_matrix)
                distance_matrix = np.where(d_update_matrix<distance_matrix,d_update_matrix,distance_matrix)
                if len(indices[0]) == 0:
                    break
            return distance_matrix

    def check_matrix(self,type_of_matrix='bo'):
        if type_of_matrix == 'adj':
            if self.get_adj_matrix() is None:
                return False
            return True
        if type_of_matrix == 'bo':
            if self.get_bo_matrix() is None:
                print ('bo matrix is not prepared!!! It is necessary to define bo matrix')
                return False
            return True

    def get_distance_matrix(self,mode = 'spatial'):
        if mode == 'spatial':
            coordinate_list = self.get_coordinate_list()
            if coordinate_list is not None:
                distance_matrix = spatial.distance_matrix(coordinate_list,coordinate_list)
            else:
                return None
        elif mode == 'graph':
            distance_matrix = self.get_matrix('distance')
        return distance_matrix                                        

    def get_ratio_matrix(self):
        atom_list = self.atom_list
        n = len(atom_list)
        radius_list = self.get_radius_list()
        radius_matrix_flatten = np.repeat(radius_list,n)
        radius_matrix = radius_matrix_flatten.reshape((n,n))
        radius_sum_matrix = radius_matrix + radius_matrix.T
        coordinate_list = self.get_coordinate_list()
        distance_matrix = spatial.distance_matrix(coordinate_list,coordinate_list)
        ratio_matrix = distance_matrix/radius_sum_matrix
        return ratio_matrix
        
    def __eq__(self,molecule):
        return self.is_same_molecule(molecule,True)
       
    def is_same_molecule(self,molecule,option=False):
        """
        Checks whether the given two molecules are the same by comparing the c_eig (see instructions in 'get_c_eig_list' within 
        class 'Molecule'
            
        :param molecule(pyclass 'Molecule):
            
        :return True/False(boolean):
            True: Two molecules are the same
            False: Two molecules are different
        """
        if len(self.atom_list) != len(molecule.atom_list):
            return False
        if option:
            if self.get_chg() != molecule.get_chg():
                return False
        c_eig_list1 = self.get_c_eig_list()
        c_eig_list2 = molecule.get_c_eig_list()
        delta_c_eig_list = np.abs(c_eig_list1-c_eig_list2)
        total_delta = np.sum(delta_c_eig_list)
        return total_delta < 1e-8 

    def copy(self,copy_all = False):
        new_molecule = Molecule()
        atom_list = self.atom_list
        # First copy atoms
        new_atom_list = []
        for atom in atom_list:
            new_atom_list.append(atom.copy())
        new_molecule.atom_list = new_atom_list
        # Copy connectivity information
        bo_matrix = self.get_matrix('bo')
        if bo_matrix is not None:
            new_molecule.bo_matrix = np.copy(bo_matrix)
        else:
            adj_matrix = self.get_matrix('adj')
            if adj_matrix is not None:
                new_molecule.adj_matrix = np.copy(adj_matrix)
            else:
                print ('Warning: Connectivity information is not included in the molecule!!!')
        
        # Finally, copy charge
        new_molecule.chg = self.get_chg()
        if 'chg' in self.atom_feature:
            new_molecule.atom_feature['chg'] = np.copy(self.atom_feature['chg'])
        # Copy molecular id 

        # Above things are essential for copy
        if copy_all:
            # Copy other attributes
            new_molecule.energy = self.energy
            new_molecule.center_of_mass = self.center_of_mass
            new_molecule.smiles = self.smiles
            new_molecule.c_eig_list = self.c_eig_list
        return new_molecule

    def get_z_list(self):
        """
        Returns atomic number list of a given molecule
         
        :param:

        :return z_list(pyclass 'numpy.ndarray'):
            list of atomic number
        """
        atom_feature = self.atom_feature
        if atom_feature is not None and 'atomic number' in atom_feature:
            return atom_feature['atomic number']
        else:
            z_list = list(map(lambda x:x.get_atomic_number(),self.atom_list))
            return np.array(z_list)

    def get_element_list(self):
        """
        Returns element list of a given molecule
         
        :param:

        :return element_list(list of string):
            list of element written as capital letters
        """
        atom_feature = self.atom_feature
        if atom_feature is not None and 'element' in atom_feature:
            return atom_feature['element']
        else:
            element_list = list(map(lambda x:x.get_element(),self.atom_list))
            return element_list

    def get_group_list(self):
        atom_feature = self.atom_feature
        if 'group' in atom_feature:
            return atom_feature['group']
        else:
            group_list = list(map(lambda x:x.get_period_group()[1],self.atom_list))
            return np.array(group_list)

    def get_period_list(self):
        atom_list = self.atom_list
        atom_feature = self.atom_feature
        if 'period' in atom_feature:
            return atom_feature['period']
        else:
            period_list = list(map(lambda x:x.get_period_group()[0],self.atom_list))
            return np.array(period_list)

    def get_period_group_list(self):
        """ 
        Returns period_group_list for atoms within molecule. See details for obtaining those period/group
        in 'get_period_group' defined within class 'Atom'
        
        :param:

        :return period_list,group_list(pyclass 'numpy.ndarray',pyclass 'numpy.ndarray'):

        """
        atom_list = self.atom_list
        atom_feature = self.atom_feature
        if 'period' in atom_feature and 'group' in atom_feature:
            return atom_feature['period'],atom_feature['group']
        n = len(atom_list)
        period_list = []
        group_list = []
        for i in range(n):
            period,group = atom_list[i].get_period_group()
            period_list.append(period)
            group_list.append(group)
        return np.array(period_list),np.array(group_list)

    def get_mass_list(self):
        atom_feature = self.atom_feature
        if 'mass' in atom_feature:
            return atom_feature['mass']
        else:
            mass_list = list(map(lambda x:x.get_mass(),self.atom_list))
            return np.array(mass_list)

    def get_chg_list(self):
        """
        Returns chg list of a given molecule
         
        :param:

        :return chg_list(pyclass 'numpy.ndarray'):
            list of formal charge
        """
        atom_feature = self.atom_feature
        try:
            return atom_feature['chg']
        except:
            return None
    
    def get_valency_list(self):
        """
        Returns the valency list of a given molecule. Valency is defined as the number of neighbors 
        (in otherwords, the number of atoms that are bonded with each other, therefore uses adj_matrix to compute valency_list)

        :param:

        :return valency_list(list of integer):
            list of valency of all atoms within a given molecule
        """
        adj_matrix = self.get_adj_matrix()
        if adj_matrix is None:
            print ('You need to define adj matrix!!!')
            return None
        valency_list = np.sum(adj_matrix,axis = 1)
        return valency_list

    def get_total_bond_order_list(self):
        """ 
        Returns the list of number of bonds for each atoms. For example of C2H4 ethene, 
        the valency of C is 3, since it is bonded to H, H, C. However, for total bond order, its value is 4, 
        since C-C is double bond. If atom order is given as [C,C,H,H,H,H], each valency, total_bond_order is given as
        valency_list: [3,3,1,1,1,1]
        total_bond_orer_list: [4,4,1,1,1,1]

        :param:

        :return total_bond_order_list(list of integer):
            list of bond order of all atoms within a given molecule

        """
        atom_list = self.atom_list
        bo_matrix = self.get_bo_matrix()
        if bo_matrix is None:
            print ('We cannot obtain bond order, since the bond order matrix is not prepared!!!')
            return None
        n = len(atom_list)        
        total_bond_order_list = np.sum(bo_matrix,axis = 1)
        return total_bond_order_list

    def get_sn_list(self):
        group_list = self.get_group_list()
        bond_sum_list = self.get_total_bond_order_list()
        chg_list = self.get_chg_list()
        return (group_list + bond_sum_list - chg_list)/2  

    def get_num_of_lone_pair_list(self):
        """
        Returns lone_pair_list for atoms within molecule
        Ex. For NH3, with atom order [N,H,H,H], the return value is [1,0,0,0]
        Ex. For H2O, with atom order [O,H,H], the return value is [2,0,0]
        :param:

        :return num_of_lone_pair_list(pyclass 'numpy.ndarray'):
            array of nonnegative integers which shows the number of lone pair(nonbonding) electrons
        """
        chg_list = self.get_chg_list()
        group_list = self.get_period_group_list()[1]
        if chg_list is None:
            print ('chg is not prepared!!! It is necessary to obtain number of lone pairs!')
            return None
        total_bond_order_list = self.get_total_bond_order_list()
        if total_bond_order_list is None:
            print ('bo matrix is not prepared!!! It is necessary to define bo matrix to obtain number of lone pairs!')
            return None
        # Now evaluate the lone pair
        num_of_lone_pair_list = (group_list - total_bond_order_list - chg_list)/2
        
        return num_of_lone_pair_list 

    def get_num_of_pi_bond_list(self):
        """
        Returns num_of_pi_bond for atoms within molecule
        Ex. For NH3, with atom order [N,H,H,H], the return value is [0,0,0,0], since there aren't any pi bonds.
        Ex. For C2H4 ethene, with atom order [C,H,H,C,H,H], the return value is [1,0,0,1,0,0]
        :param:

        :return num_of_pi_bond(pyclass 'numpy.ndarray'):
            array of nonnegative integers which shows the number of lone pair(nonbonding) electrons
        """
        bo_matrix = self.get_bo_matrix()
        if bo_matrix is None:
            print ('bo matrix is not prepared!!! It is necessary to define bo matrix to obtain the number of pi bonds!')
            return None
        pi_bond_matrix = np.where(bo_matrix > 1,bo_matrix - 1,0)
        # Only get multiple bonds
        num_of_pi_bond = np.sum(pi_bond_matrix,axis = 1)
        return num_of_pi_bond

    def get_num_of_hydrogen_list(self):
        bo_matrix = self.get_bo_matrix()
        pass

    def get_max_valency_list(self):
        """
        Returns max_valency_list for atoms within molecule. Max valency is defined in 'get_max_valency' within class 'Atom'

        :param:

        :return max_valency_list(list of integer):
            list of maximal valency of atoms within molecule 
        """
        atom_feature = self.atom_feature
        if atom_feature is not None and 'max valency' in atom_feature:
            return atom_feature['max valency']
        else:
            ### Get default max valency
            atom_list = self.atom_list
            n = len(atom_list)
            max_valency_list = []
            for i in range(n):
                atom = atom_list[i]
                max_valency_list.append(atom.get_max_valency())
            return np.array(max_valency_list)

    def get_max_bo_list(self,over_octet = False):
        atom_feature = self.atom_feature
        if atom_feature is not None and 'max bo' in atom_feature:
            return atom_feature['max bo']
        else:
            period_list = self.get_period_list()
            n = len(period_list)
            max_bo_list = 100 * np.ones((n))
            for p in range(5):
                max_bo = 1
                if over_octet:
                    pass
                else:
                    pass
                if p == 1:
                    max_bo = 4
                elif p > 1:
                    max_bo = 8
                max_bo_list = np.where(period_list == p+1,max_bo,max_bo_list)
            atom_feature['max bo'] = max_bo_list
            return max_bo_list                

    def get_octet_valence_list(self):
        atom_feature = self.atom_feature
        if atom_feature is not None and 'octet valence' in atom_feature:
            return atom_feature['octet valence']
        else:
            period_group_list = self.get_period_group_list()
            period_list = period_group_list[0]
            group_list = period_group_list[1]
            n = len(period_list)
            octet_valence_list = 100 * np.ones((n))
            for p in range(1,5):
                octet_valence = 1
                if p == 2:
                    octet_valence = 4
                elif p > 2:
                    octet_valence = group_list + 1
                octet_valence_list = np.where(period_list == p,octet_valence,octet_valence_list)
            atom_feature['octet valence'] = octet_valence_list
            return octet_valence_list

    def get_over_octet_indices(self):
        over_octet_indices = []
        period_group_list = self.get_period_group_list()
        period_list = period_group_list[0]
        group_list = period_group_list[1]
        total_bond_order_list = self.get_total_bond_order_list()
        chg_list = self.get_chg_list()
        if total_bond_order_list is None or chg_list is None:
            print ('Bond information is required!!!')
            return []
        total_valence_list = (group_list + total_bond_order_list - chg_list)/2
        n = len(period_list)
        octet_candidates = np.where(period_list > 1,4,1)
        over_octet_indices += np.where(total_valence_list > octet_candidates)[0].tolist()
        over_octet_indices.sort()
        return over_octet_indices

    def get_radius_list(self):
        """
        Returns radius list of a given molecule
        Ex. For CH4 with atom order [C,H,H,H,H], then radius_list is given as
        [0.8,0.5,0.5,0.5,0.5], if radius of C and H are given as 0.8 and 0.5 (unit is Angstrom)

        :param:

        :return radius_list(list of float):
            list of radius of each atom
        """
        atom_feature = self.atom_feature
        if 'radius' in atom_feature:
            return atom_feature['radius']
        radius_list = list(map(lambda x:x.get_radius(),self.atom_list))
        return radius_list

    def get_neighbors(self,index):
        adj_matrix = self.get_adj_matrix()
        neighbor_list = []
        n = len(self.atom_list)
        neighbors = np.where(adj_matrix[index] > 0)[0].tolist()
        neighbors.sort()
        return neighbors

    def get_neighbor_list(self):
        """
        Returns neighbor_list of a given molecule
        Ex. For CH4, with atom order [C,H,H,H,H], then neighbor list is given as
        [[1,2,3,4],[0],[0],[0],[0]] (0:C, 1,2,3,4: H)
        :param:
        
        :return neighbor_list(list of list(neighbor)):
            list(length of atom_list) of list(neighbor) 
        """
        adj_matrix = self.get_adj_matrix()
        neighbor_list = []
        n = len(self.atom_list)
        for i in range(n):
            ith_neighbors = np.where(adj_matrix[i] > 0)[0].tolist()
            ith_neighbors.sort()
            neighbor_list.append(ith_neighbors)
        return neighbor_list

    def get_neighbor_list_with_bo(self):
        atom_list = self.atom_list
        bo_matrix = self.bo_matrix
        if bo_matrix is None:
            print ('we need bond order matrix!!!')
        bond_type = [1,2,3]
        neighbor_info_list = []
        n = len(atom_list)
        for i in range(n):
            neighbor_info_list.append(dict())
        for bond_order in bond_type:
            bond_list = np.where(bo_matrix == bond_order)
            bond_list = np.stack(bond_list,axis = 1)
            for array in bond_list:
                neighbor_info_list[array[0]][array[1]] = bond_order
        return neighbor_info_list

    def get_bond_list(self,contain_bond_order = True):
        """
        Returns the total bond list as list of tuples
        For example, if CH4 is given with atom order [C,H,H,H,H], if contain_bond_order = False, the output is given as
        [(0,1),(0,2),(0,3),(0,4)]
        if contain_bond_order = True, the output is given as
        [(0,1,1),(0,2,1),(0,3,1),(0,4,1)]

        :param contain_bond_order(boolean):
            If contain_bond_order is False, it only returns bonds represented as (i,j) between atoms within the given intermediate.
            If contain_bond_order is True, it returns bonds with bond order included (i,j,bond_order), where bond_order can only have 1,2,3.
            Therefore, a given molecule(self) should be kekulized.

        :return bond_list(either list of tuple with size 2 or 3):
            bond_list
        """
        atom_list = self.atom_list
        n = len(atom_list)
        check_matrix = self.bo_matrix
        total_bond_list = []
        if contain_bond_order:
            check_matrix = self.get_matrix('bo')
            if check_matrix is None:
                contain_bond_order = False
                check_matrix = self.get_matrix('adj')
                if check_matrix is None:
                    print ('matrix',self.atom_list)
                    print ('hahahahahaha',check_matrix)
                    print ('Give connectivity! We cannot find the bond!')
                    return None
        if contain_bond_order:            
            bond_type = [1,2,3]
            check_matrix = self.get_matrix('bo')
        else:
            bond_type = [1]
            check_matrix = self.get_matrix('adj')
        for bond_order in bond_type:
            bond_list = np.where(check_matrix == bond_order)
            bond_list = np.stack(bond_list,axis = 1)
            for array in bond_list:
                if array[0] < array[1]:
                    if contain_bond_order:
                        bond_tuple = (int(array[0]),int(array[1]),int(bond_order))
                    else:
                        bond_tuple = (int(array[0]),int(array[1]))
                    total_bond_list.append(bond_tuple) 
        return total_bond_list 

    def get_trunc_bond_list(self,trunc_atom_indices,contain_bond_order = True):
        """
        Returns a subset of bond_list depending on the trunc_atom_indices. 'trunc' is originated from truncated
        For example, for CH4, you can give trunc_atom_indices a sub set of [0,1,2,3,4], ex, [0,1,2]
        Then, the result of trunc_bond_list is [(0,1),(0,2)]

        :param trunc_atom_indices(list of integer):
            Should be subset of [0,1,...,n-1], where n is the length of atom_list

        :param contain_bond_order(boolean):
            whether the return value contains the bond order information

        :return trunc_bond_list(list of tuples):
            reduced bond_list where indices of tuple only contains the indices in trunc_atom_indices
        """
        bond_list = self.get_bond_list(contain_bond_order)
        trunc_bond_list = []
        for bond in bond_list:
            atom1 = bond[0]
            atom2 = bond[1]            
            if trunc_atom_indices.count(atom1)==1 and trunc_atom_indices.count(atom2)==1:
                if atom1 < atom2:
                    if contain_bond_order:
                        bond_order = bond[2]
                        trunc_bond_list.append((atom1,atom2,bond_order))
                    else:
                        trunc_bond_list.append((atom1,atom2))
        return trunc_bond_list    

    def get_formula(self,return_type = 'dict'):
        """
        Returns stoichiometry (chemical formula) of a given molecule

        :param return_type(str):
            Mainly 'dict' and 'str' is possible
            For given water molecule, if return_type is 'dict', it returns: {'H':2,'O':1}
            If return_type is 'str'(or not 'dict'), it returns OH2 
        
        :return formula(dict or str):
            chemical formula of given molecule 

        """
        element_num = dict()
        atom_list = self.atom_list
        if atom_list is  None:
            print ('No atoms!!! We cannot get formula!')
        for atom in atom_list:
            element_type = atom.get_element()
            if element_type in element_num:
                element_num[element_type] += 1
            else:
                element_num[element_type] = 1
        if return_type == 'dict':
            return element_num
        formula = ''
        # Find increasing order
        while len(element_num) > 0:
            maximum = -100
            maximal_element_type = None
            for element_type in element_num:
                if element_num[element_type] > maximum:
                    maximum = element_num[element_type]
                    maximal_element_type = element_type
            formula += maximal_element_type+str(maximum)
            del(element_num[element_type])
        return formula

    def get_formula_as_list(self):
        """
        Returns stoichiometry (chemical formula) of a given molecule including atom indices
        For example, for given CH4 molecule with atom ordering [C,H,H,H,H], it returns dict form
        {C:[0],H:[1,2,3,4]}
        For C2H4 molecule with atom ordering [C,H,H,C,H,H], it returns dict form
        {C:[0,3],H:[1,2,4,5]}
        This function is used in arranger for evaluating chemical distance

        :param:
       
        :return element_idx_list(dict):
            chemical formula with atom indices

        """
        atom_list = self.atom_list
        element_idx_list = dict()
        if atom_list is None:
            print ('No atoms!!! We cannot get formula!')
        for i in range(len(atom_list)):
            atom = atom_list[i]
            element_type = atom.get_element()
            if element_type in element_idx_list:
                element_idx_list[element_type].append(i)
            else:
                element_idx_list[element_type] = [i]
        return element_idx_list

    def get_sssr(self):
        """
        Returns SSSR(Smallest Set of Smallest Rings) using rdkit module
        
        :param:
            
        :return(list(smallest set) of list(smallest rings)):
            list of list that contains atom indices included in the smallest rings
        """
        # Can have error with making rdkit mol
        try:
            return self.sssr
        except:
            from rdkit import Chem
            sssr = []
            skeleton_mol = self.get_skeleton() # Note: return mol is RDMol
            print (sssr)
            sssr_vectors = Chem.GetSymmSSSR(skeleton_mol)
            for sssr_vector in sssr_vectors:
                sssr.append(list(sssr_vector))
            return sssr

    def get_skeleton(self): # Sub-function for get_sssr
        from rdkit import Chem
        rd_mol = None
        valency_list = self.get_valency_list()
        n = len(self.atom_list)
        z_list = []
        for i in range(n):
            valency = valency_list[i]
            if valency == 1:
                z_list.append(1)
            elif valency <= 4:
                z_list.append(10-valency)
            else:
                z_list.append(10+valency)
        rde_mol = Chem.EditableMol(Chem.Mol())
        for i in range(n):
            rd_atom = Chem.Atom(int(z_list[i]))
            rd_atom.SetFormalCharge(0)
            rde_mol.AddAtom(rd_atom)
        bond_list = self.get_bond_list(False)
        for bond in bond_list:
            rde_mol.AddBond(bond[0],bond[1],Chem.BondType.SINGLE)
        return rde_mol.GetMol() 

    def get_minimal_data(self):
        data = dict()
        data['z'] = self.get_z_list()
        data['adj'] = self.get_adj_matrix()
        data['bo'] = self.get_bo_matrix()
        data['chg'] = self.get_chg()
        data['atom chg'] = self.get_chg_list()
        data['coords'] = self.get_coordinate_list()
        return data        

    def get_rd_mol(self,include_stereo = False):
        """
        Returns molecule with type pyclass 'rdkit.Chem.rdchem.Mol' from our type pyclass 'Molecule'
        Note that atom ordering and bond order is well preserved

        :param include_stereo(boolean):
            Do not touch this option, we have not develop options for molecule that considers stereocenter 
        
        :return rd_mol(pyclass 'rdkit.Chem.rdchem.Mol'):

        """
        from rdkit import Chem
        rd_mol = None
        bond_types = {1:Chem.BondType.SINGLE, 2:Chem.BondType.DOUBLE, 3:Chem.BondType.TRIPLE}
        rd_mol = Chem.Mol()
        rde_mol = Chem.EditableMol(rd_mol)
        atom_list = self.atom_list
        atom_feature = self.atom_feature
        chg_list = self.get_chg_list()
        n = len(atom_list)
        # Add atoms
        for i in range(n):
            atom = atom_list[i]
            rd_atom = Chem.Atom(int(atom.get_atomic_number()))
            if chg_list is not None:
                rd_atom.SetFormalCharge(int(chg_list[i]))
            rde_mol.AddAtom(rd_atom)
        # Add bonds
        bond_list = self.get_bond_list(True)
        for bond in bond_list: 
            if bond[0] < bond[1]:
                rde_mol.AddBond(bond[0],bond[1],bond_types[bond[2]])
        rd_mol = rde_mol.GetMol() 
        Chem.SanitizeMol(rd_mol)
        return rd_mol

    def get_rd_mol3D(self):
        pos = self.get_coordinate_list()
        rd_mol = self.get_rd_mol()
        Chem.SanitizeMol(rd_mol)
        params = Chem.rdDistGeom.srETKDGv3()
        params.pruneRmsThresh = 0.25
        num_sample = 20
        conformer_id_list = AllChem.EmbedMultipleConfs(rd_mol,num_sample,params)
        conformer_energy_list = []
        energy_to_id = dict()
        for conformer_id in conformer_id_list:
            converged = not AllChem.UFFOptimizeMolecule(rd_mol,confId=conformer_id)
            energy = AllChem.UFFGetMoleculeForceField(rd_mol,confId=conformer_id).CalcEnergy()
            conformer_energy_list.append(energy)
            energy_to_id[energy] = conformer_id
        
        if len(conformer_energy_list) == 0:
            print ('No conformer is generated !!!')
            print (self.get_smiles())
            return None

        conformer_energy_list.sort()
        energy = conformer_energy_list[0]
        conformer_id = energy_to_id[energy] 
        try:
            conformers = rd_mol.GetConformers()
        except:
            print ('No conformer for',self.get_smiles('ace'))
            return None
        conformer = conformers[conformer_id]
        for i in range(rd_mol.GetNumAtoms()):
            xyz = (float(pos[i][0]), float(pos[i][1]), float(pos[i][2]))
            conformer.SetAtomPosition(i, xyz)
        return rd_mol

    def get_ob_mol(self,include_stereo=False):
        """
        Returns molecule with type pyclass 'openbabel.OBMol' from our type pyclass 'Molecule'
        Note that atom ordering and bond order is well preserved
        :param include_stereo(boolean):
            Do not touch this option, we have not develop options for molecule that considers stereocenter 
        
        :return rd_mol(pyclass 'openbabel.OBMol'):
        """
        from openbabel import openbabel

        ob_mol = openbabel.OBMol()
        atom_list = self.atom_list
        bond_list = self.get_bond_list(True)
        n = len(atom_list)
        z_list = self.get_z_list()
        chg_list = None
        atom_feature = self.atom_feature
        if atom_feature is not None and 'chg' in atom_feature:
            chg_list = atom_feature['chg']
        # Generate atoms
        for i in range(n):
            atom = atom_list[i]
            ob_atom = ob_mol.NewAtom()
            z = int(atom.get_atomic_number())
            ob_atom.SetAtomicNum(z)
            ob_atom.SetFormalCharge(int(chg_list[i]))
        # Generate bonds
        for bond in bond_list:
            ob_mol.AddBond(bond[0]+1,bond[1]+1,bond[2])                
        '''
        else:
            import pybel
            ob_mol = pybel.readstring('smi',self.smiles)
        '''
        return ob_mol

    def get_coordinate_list(self):
        """
        Returns 3d coordinate of a given molecule
        
        :param:

        :return coordinate_list(list(size n) of tuple of float(size 3)):
            
        """
        coordinate_list = []
        atom_list = self.atom_list
        if 'coords' in self.atom_feature:
            return self.atom_feature['coords']
        for atom in atom_list:
            coordinate_list.append([atom.x,atom.y,atom.z])
        return np.array(coordinate_list)

    def print_coordinate_list(self,option='element'):
        coordinate_list = self.get_coordinate_list()
        atom_list = self.atom_list
        n = len(atom_list)
        for i in range(n):
            coordinate = coordinate_list[i]
            element = atom_list[i].get_element()
            if option == "number":
                element = atom_list[i].get_atomic_number()
            print_x = f"{coordinate[0]:>12.8f}"
            print_y = f"{coordinate[1]:>12.8f}"
            print_z = f"{coordinate[2]:>12.8f}"
            print(f"{element:<3} {print_x} {print_y} {print_z}")

    def sanitize(self):
        adj_matrix = self.get_adj_matrix()
        if adj_matrix is None:
            print ('Cannot sanitize molecule because there is no adj matrix information !!!')
        else:
            chg = self.get_chg()
            if chg is None:
                print ('Cannot sanitize molecule because there is no charge information !!!')
            bo_matrix = process.get_bo_matrix_from_adj_matrix(self,chg)
            fc_list = process.get_chg_list_from_bo_matrix(self,chg,bo_matrix)
            self.bo_matrix = bo_matrix
            self.atom_feature['chg'] = fc_list

    def get_valid_molecule(self,remove_radical = True,method = 'pulp',**kwargs):
        z_list = self.get_z_list()
        original_z_list = np.copy(z_list)
        n = len(z_list)
        chg = self.get_chg()
        if chg is None:
            virtual_chg = 0
        else:
            virtual_chg = chg
        period_list, group_list = self.get_period_group_list()
        adj_matrix = self.get_adj_matrix()
        adjacency_list = np.sum(adj_matrix, axis=0)

        #Compute SN
        problem_indices = np.flatnonzero(adjacency_list > self.get_max_valency_list())
        over_period_indices = np.flatnonzero(period_list > 3)

        # Add problematic indices for over period indices ...
        problem_indices = list(set(problem_indices) | set(over_period_indices))
        new_z_list = np.copy(z_list)

        for idx in problem_indices:
            period = period_list[idx]
            group = group_list[idx]
            adj = adjacency_list[idx]
            if period == 1:
                if adj > 1:
                    new_z_list[idx] = 10 - adj
            else:
                if adj < 5:
                    if period == 2:
                        new_z_list[idx] = 10 - adj
                    else:
                        new_z_list[idx] = 18 - adj
                elif adj < 7:
                    new_z_list[idx] = 10 + adj #replace with one higher period element with proper valency
                else:
                    new_z_list[idx] = 26 #replace with Fe
        
        #Construct new Molecule
        virtual_molecule = Molecule([new_z_list, adj_matrix, None, None])

        #Construct BO and Chg
        new_period_list, new_group_list = virtual_molecule.get_period_group_list()

        if method == 'pulp':
            new_chg_list, new_bo_matrix = process.get_chg_list_and_bo_matrix_pulp(virtual_molecule, virtual_chg, **kwargs)
            if new_bo_matrix is None: # If solver failed to solve ...
                return None
            else:
                not_valid = False
                new_bo_sum = np.sum(new_bo_matrix, axis=0)
                # Recheck overvalence ...
                for i in range(n):
                    if new_period_list[i] == 1:
                        if new_bo_sum[i] > 1:
                            reconstruct = True
                            break
                    if new_period_list[i] == 2:
                         if new_bo_sum[i] > 4:
                             reconstruct = True
                             break
                if not_valid:
                    return None
        else:
            new_bo_matrix = process.get_bo_matrix_from_adj_matrix(virtual_molecule, virtual_chg)

        new_bo_sum = np.sum(new_bo_matrix, axis=0)
        new_period_list, new_group_list = virtual_molecule.get_period_group_list()
        # Modify new_z for atoms containing unpaired electron ...
        for i in range(n):
            group = new_group_list[i]
            bo = new_bo_sum[i]
            if remove_radical and group % 2 != bo % 2:
                new_group_list[i] += 1
                virtual_molecule.atom_list[i].atomic_number += 1

        for i in range(n):
            group = new_group_list[i]
            bo = new_bo_sum[i]
            parity = 0
            # new_z: Modified z when overvalence is observed. = 10-bo for period=2, = 18 - bo (bo <=4), = bo + 10 (bo>4)
            if new_period_list[i] == 1:
                octet = 1 # For checking validity
                new_z = 10 - bo
            elif new_period_list[i] == 2:
                octet = min(group,4)
                new_z = 10 - bo
            else:
                octet = group
                if bo > 4 and bo < 7:
                    new_z = bo + 10 # Reconsider valence expansion
                elif bo >= 7:
                    new_z = 26
                else:
                    new_z = 18 - bo # Just same with F, O, N substitution, but with higher order
            if not process.check_atom_validity(group,bo,0,octet): # Set every charge equal to zero
                virtual_molecule.atom_list[i].set_atomic_number(int(new_z))
        virtual_molecule.set_bo_matrix(new_bo_matrix)
        virtual_molecule.atom_feature['chg'] = np.zeros((n)) # Set charge zero
        new_z_list = virtual_molecule.get_z_list()
        return virtual_molecule

    def make_3d_coordinate(self,library='rdkit'):
        return self.make_3d_coordinates(1,library)[0]

    def make_3d_coordinates(self,num_conformer = 1,library = 'rdkit'):
        """
        Returns possible 3d molecular geometry using other libraries, mainly 'babel' and 'rdkit'

        :param library(str):
            Either 'babel' or 'rdkit' are possible. You can add your own found libraries for generating 3d structure 
        
        :return coordinate_list(list of tuples with size 3(float)):
            3d geometry of molecule generated by other libraries

        """
        mol = None
        use_ic_update = False
        coordinates = []
        # Make permutations depending num_conformer
        z_list = self.get_z_list()
        num_sample = 20
        n = len(z_list)
        if library == 'rdkit':
            params = Chem.rdDistGeom.srETKDGv3()
            params.pruneRmsThresh = 0.25
            try:
                mol = self.get_rd_mol()
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
            except:
                virtual_molecule = self.get_valid_molecule()
                mol = virtual_molecule.get_rd_mol()
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
                use_ic_update = True
            if mol is None:
                print ('Impossible embedding')
                return []
            conformer_id_list = AllChem.EmbedMultipleConfs(mol, num_sample, params)
            conformer_energy_list = []
            energy_to_id = dict()
            for conformer_id in conformer_id_list:
                try:
                    converged = not AllChem.UFFOptimizeMolecule(mol,confId=conformer_id)
                    energy = AllChem.UFFGetMoleculeForceField(mol,confId=conformer_id).CalcEnergy()
                    conformer_energy_list.append(energy)
                    energy_to_id[energy] = conformer_id
                except:
                    continue
            if len(conformer_energy_list) == 0:
                print ('No conformer is generated !!!')
                print (self.get_smiles())
                return []
            conformer_energy_list.sort()
            conformers = mol.GetConformers()
            for energy in conformer_energy_list[:num_conformer]:
                conformer_id = energy_to_id[energy]
                conformer = conformers[conformer_id]
                coordinate_list = []
                for i in range(n): 
                    position = conformer.GetAtomPosition(i) 
                    coordinate_list.append((position[0],position[1],position[2]))
                if len(coordinate_list) > 0:
                    coordinate_list = np.array(coordinate_list)
                    coordinates.append(coordinate_list)
        elif library == 'babel': 
            from openbabel import pybel
            from openbabel import openbabel 
            import os
            #### pybel method
            try:
                ob_mol = self.get_ob_mol()
            except:
                virtual_molecule = self.get_valid_molecule()
                ob_mol = virtual_molecule.get_ob_mol()
                use_ic_update = True
            pybel_mol = pybel.Molecule(ob_mol)
            for i in range(num_conformer):
                coordinate_list = []
                pybel_mol.make3D()
                pybel_mol.localopt('uff',1000)
                pybel_atom_list = pybel_mol.atoms
                for atom in pybel_atom_list:
                    position = atom.coords
                    coordinate_list.append((position[0],position[1],position[2]))
                if len(coordinate_list) > 0:
                    coordinates.append(coordinate_list)
        else:
            print ('Give us algorithm for generating 3D!!!')
            print ('You can try your own algorithm here!!!')
            ######### Your own algorithm here #########
            return None
        # (TODO): Need to come up with better algorithm, best one is to simply reoptimize with uff
        # Reupdate molecule
        scale = 1.0 # Make distance to bond length, value between 1.05 ~ 1.2
        # Repermute coordinates
        for i in range(len(coordinates)):
            coordinate_list = coordinates[i]
            if use_ic_update:
                internal_coordinates = self.get_bond_list(False)
                q_updates = dict()
                radius_list = self.get_radius_list()
                for bond in internal_coordinates:
                    delta_d = scale * (radius_list[bond[0]] + radius_list[bond[1]]) - ic.get_distance(coordinate_list,bond[0],bond[1])
                    if delta_d < -0.15:
                        q_updates[bond] = delta_d
                    else:
                        q_updates[bond] = 0.0
                ic.update_xyz(coordinate_list,q_updates)
            coordinates[i] = coordinate_list
        return coordinates

    def sample_conformers(self,n_conformer = 20,library = 'rdkit',rmsd_criteria = 1.0):
        conformer_list = []
        coordinates = self.make_3d_coordinates(n_conformer,library)
        for coordinate_list in coordinates:
            ace_mol = Molecule((self.get_z_list(),self.get_adj_matrix(),None,self.get_chg_list()))
            process.locate_molecule(ace_mol,coordinate_list)
            # Check validity of molecule geometry
            if not ace_mol.is_appropriate_geometry():
                continue
            # Check RMSD between conformers
            put_in = True
            for i,conformer in enumerate(conformer_list):
                rmsd = process.get_rmsd(conformer,ace_mol)
                if rmsd < rmsd_criteria:
                    put_in = False
                    break
            if put_in:
                ace_mol.chg = self.get_chg()
                conformer_list.append(ace_mol)
        return conformer_list

    def write_geometry(self,file_directory, option='element',criteria = 1e-4):
        """
        Writes xyz file that contains the 3d molecular geometry
    
        :param file_directory(str):
            Directory for saving 3d geometry xyz file

        :return:
        """
        atom_list = self.atom_list
        n = len(atom_list)
        f = open(file_directory, 'w')
        if True: # If inappropriate geometry condition is determined, it will be added
            content = str(n)+'\n'            
            if self.energy is not None:
                content = content + str(self.energy)+'\n'
            else:
                content = content + '\n'
            f.write(content)
            for atom in atom_list:
                f.write(atom.get_content(option,criteria))
            f.close()
        else:
            print ('Wrong geometry!!!')

    def get_content(self,option='element',criteria=1e-4):
        atom_list = self.atom_list
        n = len(atom_list)
        content = str(n)+'\n'            
        if self.energy is not None:
            content = content + str(self.energy)+'\n'
        else:
            content = content + '\n'
        for atom in atom_list:
            content = content + atom.get_content(option,criteria)
        return content

    def save_as_pickle(self,file_directory):
        """
        Save object with least data as following (dict)
        'z':z_list, 'adj': adj_matrix, 'bo': bo_matrix, 'chg': chg_list, 'coords': coordinate_list 
        """
        import pickle
        data = self.get_minimal_data()
        with open(file_directory,'wb') as f:
            pickle.dump(data,f)

    def write_input(self,save_directory,template_directory=None,parameters={'mult':None,'chg':None}):
        use_ase = False
        if use_ase:
            import ase
            ase_atoms = self.get_ase_atoms()
            ase.io.write(save_directory,ase_atoms)
            return 
        input_file = open(save_directory,'w')
        if template_directory is None:
            print ('Warning: Generated input is for running DFT calculation with Gaussian!!!')
            input_file.write('#N B3LYP/6-31G(d) SP\n\nTitle\n\n')
        else:
            f = open(template_directory)
            for line in f:
                input_file.write(line)
            f.close()
        if save_directory[-4:]!='.xyz':
            import os
            print ('Warning: Filename extension is not given!!! We used .com extension')
            save_directory = os.path.join(save_directory,'tmp.com')
        if parameters['mult'] == None:
            mult = (-self.chg + int(np.sum(self.get_z_list())))%2
            mult = int(2*mult+1)
        if parameters['chg'] == None:
            chg = int(self.chg)
        input_file.write(str(chg) + ' ' + str(mult)+'\n')
        # All specified
        atom_list = self.atom_list 
        coordinate_list = self.get_coordinate_list()
        if coordinate_list is None:
            coordinate_list = self.make_3d_coordinate()
        if coordinate_list is None or len(coordinate_list) == 0:
            print ('No molecule is generated!!!!')
            return
        for i,coordinate in enumerate(coordinate_list):
            x = str(coordinate[0])
            y = str(coordinate[1])
            z = str(coordinate[2])
            input_file.write(atom_list[i].get_element()+' '+x+' '+y + ' '+z + '\n')
        input_file.write('\n\n')

    def write_uff_input(self,save_directory):
        chg = int(self.chg)
        mult = (-self.chg + int(np.sum(self.get_z_list())))%2
        mult = int(2*mult+1)
        input_file = open(save_directory,'w')
        if save_directory[-4:]!='.xyz':
            import os
            save_directory = os.path.join(save_directory,'tmp.com')
        input_file.write('#N opt uff=qeq geom=connectivity iop(2/12=1, 4/120=100000)\n\nTitle\n\n')
        input_file.write(str(chg) + ' ' + str(mult)+'\n')
        # All specified
        atom_list = self.atom_list
        coordinate_list = self.make_3d_coordinate()
        if coordinate_list is None or len(coordinate_list) == 0:
            print ('No molecule is generated!!!!')
            return
        for i,coordinate in enumerate(coordinate_list):
            x = str(coordinate[0])
            y = str(coordinate[1])
            z = str(coordinate[2])
            input_file.write(atom_list[i].get_element()+' '+x+' '+y + ' '+z + '\n')
        input_file.write('\n\n')

    def is_appropriate_geometry(self,criteria = 0.5):
        """
        Returns whether the current molecule has appropriate geometry
        This is done by checking whether two atoms are too close each other. 
        Here, we set the criteria for too close two atoms as 0.5

        :param criteria(float):
            The criteria for recognizing whether the two atoms are too close

        :return True/False:
            True: The geometry is appropriate
            False: The geometry is not appropriate

        """
        atom_list = self.atom_list
        coordinate_list = self.get_coordinate_list()
        return process.check_geometry(coordinate_list,criteria)

    def get_normal_vector(self,idx1,idx2,idx3):
        """
        For given three atom indices, it returns normal vector that is perpendicular to
        the plane generated by three indices

        :param idx1,idx2,idx3(int):
            Indices for selected three atoms

        :return normal_vector(pyclass 'numpy.ndarray' with length 3)
            normal vector
        """
        vector1 = self.get_vector_between_atoms(idx3,idx1)
        vector2 = self.get_vector_between_atoms(idx3,idx2)
        cross_vector = np.cross(vector1,vector2)
        norm = np.linalg.norm(cross_vector)
        if norm == 0: 
            return cross_vector
        else:
            return cross_vector/norm
   
    def get_vector_between_atoms(self,idx1,idx2,normalize = True):
        """
        For given two atom indices, it returns difference vector between two atoms. This function can be used to 
        move or rotate a given molecule
        
        :param idx1,idx2(int):
            Indices for selected two atoms 

        :return difference_vector(pyclass 'numpy.ndarray' with length 3)

        """

        """
        something was wrong here.
        changed in Jul 7th, 2020
        - Jinwon Lee

        previous state
        atom_list = self.atom_list
        atom_coord1 = self.get_coordinate()
        atom_coord2 = self.get_coordinate()
        """

        atom_list = self.atom_list
        atom_coord1 = atom_list[idx1].get_coordinate()
        atom_coord2 = atom_list[idx2].get_coordinate()
        vector = atom_coord2 - atom_coord1
        if normalize:
            norm = np.linalg.norm(vector)
            if norm < 0.0001:
                print ('zero vector is found ...')
                return vector
            else:
                return vector/norm
        return vector

    def get_internal_coordinate(self,indices,unit='degree'):
        if len(indices) == 2:
            idx1,idx2 = indices
            return self.get_distance_between_atoms(idx1,idx2)
        elif len(indices) == 3:
            idx1, idx2, idx3 = indices
            return self.get_angle_between_atoms(idx1,idx2,idx3,unit)
        elif len(indices) == 4:
            idx1, idx2, idx3, idx4 = indices
            return self.get_dihedral_angle_between_atoms(idx1,idx2,idx3,idx4,unit)
        else:
            print (f'Wrong coordinate (={indices}) given!')
            return None

    def get_distance_between_atoms(self,idx1,idx2):
        """
        Returns the distance between chosen two atoms

        :param idx1,idx2(int):
            indices of chosen two atoms. 

        :return distance(float):
            Distance between selected two atoms

        """
        coordinate_list = self.get_coordinate_list()
        return ic.get_distance(coordinate_list,idx1,idx2)

    def get_angle_between_atoms(self,idx1,idx2,idx3,unit='rad'):
        """
        Returns the distance between chosen two atoms

        :param idx1,idx2(int):
            indices of chosen two atoms. 

        :return distance(float):
            Distance between selected two atoms

        """
        coordinate_list = self.get_coordinate_list()
        angle = ic.get_angle(coordinate_list,idx1,idx2,idx3)
        if unit == 'degree':
            angle *= 180/np.pi
        return angle


    def get_dihedral_angle_between_atoms(self,idx1,idx2,idx3,idx4,unit='rad'):
        coordinate_list = self.get_coordinate_list()
        angle = ic.get_dihedral_angle(coordinate_list,idx1,idx2,idx3,idx4)
        if unit == 'degree':
            angle *= 180/np.pi
        return angle

    def get_molecule_radius(self):
        """
        get molecule radius(distance from CM to farthest atom)

        :param self
            class Molecule.

        :return:
        radius: float. molecule radius


        --Jinwon Lee
        """
        center = self.get_center()
        radius = 0
        for atom in self.atom_list:
            tmp = np.linalg.norm(center - atom.get_coordinate()) + atom.get_radius();
            if radius < tmp:
                radius = tmp
        return radius

    def draw_with_indices(self, file_dir=None, change_indices=None):
        from rdkit.Chem import Draw
        # Produces an image with indices
        # if file_dir is provided, the image is saved 
        # if file_dir is not provided, only shows the image without saving
        rd_mol = self.get_rd_mol()
        for atom in rd_mol.GetAtoms():
            if change_indices is None:
                atom.SetAtomMapNum(atom.GetIdx())
            elif change_indices[atom.GetIdx()] != -1:
                atom.SetAtomMapNum(int(change_indices[atom.GetIdx()]))
        if file_dir is None:
            Draw.ShowMol(rd_mol, size=(600, 600))
        else:
            Draw.MolToFile(rd_mol, file_dir, size=(600, 600))

    def visualize(self,method='rdkit'):
        # If it already has optimized geometry, use those geometires
        coordinate_list = self.get_coordinate_list()
        count_zeros = 0
        for coordinate in coordinate_list:
            d = np.linalg.norm(np.array(coordinate))
            if d < 0.001:
                count_zeros += 1
        if count_zeros > 1:
            coordinate_list = self.make_3d_coordinate(method)
            process.locate_molecule(self,coordinate_list)
        self.write_geometry('tmp.xyz')
        import os
        os.system('molden tmp.xyz')  
        os.system('rm tmp.xyz')

    def get_intermediate(self):
        data = [self.get_z_list(),self.get_adj_matrix(),self.get_bo_matrix(),self.get_chg_list()]
        intermediate = Intermediate(data)
        return intermediate


class Intermediate(Molecule):
    """
    :class Intermediate:
        class Intermediate mainly contains atom_list, atom_feature, bo_matrix, adj_matrix, energy, smiles
        which are the same thing in class Molecule
        class Intermediate is the set of molecules. Thus, it also has two more variables: molecule_list, a set of molecule and atom_indices_for_each_molecule
        which corresponds to the actual indices within Intermediate for each atom_list in molecule_list
        For example, if intermediate contains two H2 molecules, then one example for atom_indices_for_each_molecule is
        "[[0,1],[2,3]]" or "[[0,2],[1,3]]"
        number_list is **** (need change)

    :param data(various types are possible):
        data that represents Intermediate is used
        Possible input for data: 'CCC.CCC', list of pyclass Molecule

    : param data_type(string):
         the data type for generating class Intermediate. Mainly 'smiles','molecule_list' are possible input
         when 'smiles' is used, it generates molecule using RDkit within class Molecule
         when 'molecule_list' is used, it directly generates Intermediate, by appending molecule in molecule_list
         when None is used, Intermediate with an empty data is generated.
    """
    def __init__(self,data = None):
        molecule_list = None
        if type(data) == str:
            super().__init__(data) # If smiles or xyz -> require atom_indices_for_each_molecule
        elif type(data) is list:
            try:
                len(data[0]) # If data[0] is z_list -> require atom_indices_for_each_molecule
                super().__init__(data)
            except:
                super().__init__(None)
                molecule_list = data # Only molecule_list is formed when molecule_list is given as input ...
        elif data is None:
            super().__init__(None)
        else:
            print ('Unknown input type!!!')
            pass
        self.number_list = []
        self.molecule_list = None
        self.intermediate_id = None
        self.atom_indices_for_each_molecule = None
        if molecule_list is not None: # Make every thing ... atom_list, adj, bo, chg
            # First, obtain atom indices
            atom_list = []
            cnt = 0
            atom_feature = dict()
            # From molecules obtain information: atom indices (atom_list), atom_feature,
            for molecule in molecule_list:
                atom_indices = []
                molecule_atom_list = molecule.atom_list
                molecule_atom_feature = molecule.atom_feature
                m = len(molecule_atom_list)
                # Generate atom list
                for i in range(m):
                    atom_list.append(molecule_atom_list[i])
                    atom_indices.append(i+cnt)
                # Generate key feature
                if molecule_atom_feature is not None:
                    for key_feature in molecule_atom_feature:
                        if key_feature in atom_feature:
                            atom_feature[key_feature] = np.concatenate((atom_feature[key_feature],molecule_atom_feature[key_feature]))
                        else:
                            atom_feature[key_feature] = molecule_atom_feature[key_feature]
                cnt += m
            total_bo_matrix = np.zeros((cnt,cnt))
            total_adj_matrix = np.zeros((cnt,cnt))
            cnt = 0
            # Now, generate molecule adj/bo matrix using slice
            update_bo_matrix = True
            update_adj_matrix = True
            for molecule in molecule_list:
                m = len(molecule.atom_list)
                bo_matrix = molecule.bo_matrix
                if bo_matrix is None:
                    update_bo_matrix = False
                    # Check adj
                    adj_matrix = molecule.adj_matrix
                    if adj_matrix is None:
                        print ('Molecule connectivity is not given!!!!')
                        update_adj_matrix = False
                    else:
                        total_adj_matrix[cnt:cnt+m,cnt:cnt+m] = adj_matrix[:,:]
                else:
                    total_bo_matrix[cnt:cnt+m,cnt:cnt+m] = bo_matrix[:,:]
                    total_adj_matrix = np.where(total_bo_matrix>0,1,0)
                cnt += m 
            if update_bo_matrix:
                self.bo_matrix = total_bo_matrix
            if update_adj_matrix:
                self.adj_matrix = total_adj_matrix
            self.atom_feature = atom_feature
            self.atom_list = atom_list
                    
        # Update self.chg
        if 'chg' in self.atom_feature:
            self.chg = np.sum(self.atom_feature['chg'])

    def setCeig(self,Ceig):
        self.Ceig = Ceig

    def set_name(self,name): # Function for ACE-Reaction
        self.name = name

    def get_atom_indices_for_each_molecule(self,update = False):
        """
        Computes the atom_index for each atom within molecules. The example of atom_indices_for_each_molecule is provided
        at __init__ section.
        
        :param:
            
        :return atom_indices_for_each_molecule(list of list):
            list of atom_indices for pyclass Intermediate for each molecule (Example shown in __init__)
        """ 
        if self.atom_indices_for_each_molecule is not None:
            return self.atom_indices_for_each_molecule
        index = 0
        n = len(self.atom_list)
        atom_indices_for_each_molecule = []
        if n > 1:
            adj_matrix = self.get_matrix('adj')
            if adj_matrix is None:
                return None
            atom_indices_for_each_molecule = process.group_molecules(adj_matrix)
            if update:
                self.atom_indices_for_each_molecule = atom_indices_for_each_molecule
            return atom_indices_for_each_molecule
        else:
            return [[0]]

    def get_molecule_list(self,update = False):
        """
        Returns the molecule_list stored in pyclass Intermediate
        
        :param:
            
        :return molecule_list(list of pyclass 'Molecule'):
            list of class 'Molecule'. If the 'Intermediate' does not contain molecule_list as variable, the method
            finds molecule_list if the atom_list and adjacency_matrix of Intermediate are provided
        """
        if self.molecule_list is not None:
            return self.molecule_list
        else:
            molecule_list = []
            atom_indices_for_each_molecule = self.get_atom_indices_for_each_molecule()
            for atom_indices in atom_indices_for_each_molecule:
                molecule = self.get_molecule_from_indices(atom_indices)
                molecule_list.append(molecule)
            if update:
                self.molecule_list = molecule_list
            return molecule_list

    def get_molecule_from_indices(self,indices):
        """
        Returns pyclass 'Molecule' for corresponding indices within pyclass 'Intermediate'
        Ex. [1,2,3,4] is given, with atom_list([N atom,H atom, H atom, H atom, O atom, H atom, H atom]), it returns 
        class 'Molecule', corresponding to NH3, 
        
        :param indices(list of int):
        indices of atoms within atom_list of class 'Intermediate' 
        
        :return molecule(pyclass 'Molecule'):
        A pyclass 'Molecule' whose variables that can be copied from class 'Intermediate' are copied
        """
        atom_list = self.atom_list
        adj_matrix = self.adj_matrix
        bo_matrix = self.bo_matrix
        atom_feature = self.atom_feature
        
        #### Get molecule feature
        n = len(indices)
        molecule_atom_list = []
        molecule_atom_feature = dict()
        molecule = Molecule()
        
        # Generate atom_list
        for indice in indices:
            molecule_atom_list.append(atom_list[indice])
            molecule.atom_list = molecule_atom_list
        # Generate atom feature
        for key_feature in atom_feature:
            if atom_feature[key_feature] is not None:
                molecule_atom_feature[key_feature] = atom_feature[key_feature][indices]
            else:
                molecule_atom_feature = None
            molecule.atom_feature = molecule_atom_feature
        
        # Update charge if possible ...
        if molecule.atom_feature is not None and 'chg' in molecule.atom_feature:
            molecule.chg = np.sum(molecule.atom_feature['chg'])
        
        # Generate bo_matrix, if exists
        if bo_matrix is None:
            # Check adj matrix
            if adj_matrix is None:
                print ('We need connectivity to get information of desired molecule!!!')
            else:
                reduced_adj_matrix = adj_matrix[:,indices]
                reduced_adj_matrix = reduced_adj_matrix[indices,:]
                molecule.adj_matrix = reduced_adj_matrix
        else:
            reduced_bo_matrix = bo_matrix[:,indices]
            reduced_bo_matrix = reduced_bo_matrix[indices,:]
            molecule.bo_matrix = reduced_bo_matrix
        return molecule

    def get_multiplicity(self):
        if self.multiplicity is None:
            molecule_list = self.get_molecule_list()
            num_spin = 0
            for molecule in molecule_list:
                num_spin += (molecule.get_multiplicity()-1)
            return (num_spin + 1)
        else:
            return int(self.multiplicity)

    def __eq__(self,intermediate):
        return self.is_same_intermediate(intermediate,True)

    def is_same_intermediate(self,intermediate,option = False):
        """
        Checks whether the two intermediates are same or not. 
        The method checks whether the molecules within molecule_list are the same
                 
        :param intermediate(pyclass 'Intermediate'):
            molecule_list of intermediates are not necessarily needed to be specified. The method automatically 
            checks molecule_list and computes molecule_list if the molecule_list is not provided.

        :return True/False (boolean):
            If True, they are the same. Otherwise, different
        """
        if len(self.atom_list) != len(intermediate.atom_list):
            return False
        molecule_list1 = self.get_molecule_list()
        molecule_list2 = intermediate.get_molecule_list()
        if molecule_list1 == None or molecule_list2 == None:
            print ('intermediate is not well prepared!')
            return False
        elif len(molecule_list1) != len(molecule_list2):
            return False
        n = len(molecule_list1)
        molecule_indices1 = list(range(n))
        molecule_indices2 = list(range(n))
        cnt = n
        while cnt > 0:
            molecule = molecule_list1[molecule_indices1[0]]
            found = False
            for i in range(cnt):
                index = molecule_indices2[i]
                molecule_prime = molecule_list2[index]
                if molecule.is_same_molecule(molecule_prime,option):
                    del(molecule_indices1[0])
                    del(molecule_indices2[i])
                    cnt -= 1
                    found = True
                    break
            if not found:
                break
        return (cnt == 0)

    def copy(self,copy_all = False):
        new_intermediate = Intermediate()
        atom_list = self.atom_list
        # First copy atoms
        new_atom_list = []
        for atom in atom_list:
            new_atom_list.append(atom.copy())
        new_intermediate.atom_list = new_atom_list
        # Copy connectivity information
        bo_matrix = self.get_matrix('bo')
        if bo_matrix is not None:
            new_intermediate.bo_matrix = np.copy(bo_matrix)
        else:
            adj_matrix = self.get_matrix('adj')
            if adj_matrix is not None:
                new_intermediate.adj_matrix = np.copy(adj_matrix)
            else:
                print ('Warning: Connectivity information is not included in the molecule!!!')

        #new_intermediate.atom_indices_for_each_molecule = self.get_atom_indices_for_each_molecule()
        # Finally, copy charge
        chg = self.get_chg()
        new_intermediate.chg = chg 
        if 'chg' in self.atom_feature:
            new_intermediate.atom_feature['chg'] = np.copy(self.atom_feature['chg'])
        # Above things are essential for copy
        if copy_all:
            # Copy other attributes
            new_intermediate.energy = self.energy
            new_intermediate.center_of_mass = self.center_of_mass
            new_intermediate.smiles = self.smiles
            new_intermediate.c_eig_list = self.c_eig_list
        return new_intermediate

    def initialize(self,method='SumofFragments'): # Make new bo/chgs/atom_indices_for_each_molecule if possible ...
        atom_indices_for_each_molecule = self.get_atom_indices_for_each_molecule()
        n = len(self.atom_list)
        bo_matrix = np.zeros((n,n))
        chg_list = np.zeros((n))
        for atom_indices in atom_indices_for_each_molecule:
            molecule = self.get_molecule_from_indices(atom_indices)
            molecule_chg = molecule.get_chg()
            if molecule_chg is None:
                print ('Cannot initialize molecule !')
                return
            mol_chg_list, mol_bo_matrix = process.get_chg_and_bo(molecule,molecule_chg,method=method)
            chg_list[atom_indices] = mol_chg_list
            index_function = np.ix_(atom_indices,atom_indices)
            bo_matrix[index_function] = mol_bo_matrix
        self.atom_feature['chg'] = chg_list
        self.bo_matrix = bo_matrix

    def get_formula_id(self):
        if self.formula_id is not None:
            return self.formula_id
        formula_id = []
        for molecule in self.get_molecule_list():
            formula_id.append(molecule.get_formula_id())
        formula_id.sort()
        formula_id = tuple(formula_id)
        return formula_id

    def get_intermediate_id(self):
        if self.intermediate_id is not None:
            return self.intermediate_id
        molecule_list = self.get_molecule_list()
        if molecule_list is None:
            return None
        else:
            intermediate_id = []
            for molecule in molecule_list:
                molecule_id = molecule.get_molecule_id()
                if molecule_id is None:
                    return None
                intermediate_id.append(molecule_id)
            intermediate_id.sort()
            return tuple(intermediate_id)

    def get_smiles(self,method = 'SumofFragments',find_stereocenter = 'N'):
        """ 
        Returns the smiles of molecules connected with .
        For example, if two methane molecules are given in the intermeidate, 
        it returns 'C.C'
        
        :param:
            
        :return smiles(str):
            Total smiles where each molecule is split by using '.' 
        """
        adj_matrix = self.get_adj_matrix()
        fc_list = self.get_chg_list()
        if adj_matrix is None or fc_list is None:
            print ('We need to know both adjacency and charge!!!')
            return None
        molecule_list = self.get_molecule_list()
        if molecule_list is None:
            print ('Cannot make SMILES, as molecule list cannot be well created !!!')
        smiles_list = []
        for molecule in molecule_list:
            smiles_list.append(molecule.get_smiles(method,find_stereocenter))
        if smiles_list[0] is None:
            print ('SMILES is not well created !!!')
            smiles = None
        else:
            smiles = '.'.join(smiles_list)
        return smiles

    def get_energy(self):
        if self.energy is None:
            energy = 0
            molecule_list = self.get_molecule_list()
            for molecule in molecule_list:
                if molecule.energy is None:
                    return None
                else:
                    energy += molecule.energy
            return energy
        else:
            return self.energy

    def make_3d_coordinates(self,num_conformer = 1,library = 'rdkit'):
        grid = 2.0
        coordinates = super().make_3d_coordinates(num_conformer,library)
        atom_indices_for_each_molecule = self.get_atom_indices_for_each_molecule()
        if atom_indices_for_each_molecule is None:
            return []
        geometry_info = dict()
        atom_list = self.atom_list
        n = len(atom_indices_for_each_molecule)
        translation_vector = dict()
        # Make sphere for each molecule
        for coordinate_list in coordinates:
            for atom_indices in atom_indices_for_each_molecule:
                center = np.mean(coordinate_list[atom_indices],axis=0)
                radius = 0
                for atom_index in atom_indices:
                    r = np.linalg.norm(center - coordinate_list[atom_index]) + atom_list[atom_index].get_radius() * 2
                    if r > radius:
                        radius = r
                geometry_info[tuple(atom_indices)] = [center,radius]
            if len(geometry_info) == 1:
                continue
            # Scatter molecules
            index = 0
            while index < n:
                # Grid search for molecule
                overlap = True
                l = 0
                current_atom_indices = tuple(atom_indices_for_each_molecule[index])
                current_center,current_radius = geometry_info[current_atom_indices]
                good_combination = None
                while overlap:
                    # For every grid, check overlap
                    combinations = []
                    for x in range(l+1):
                        for y in range(l+1-x):
                            combinations.append((x,y,l-x-y))
                    for combination in combinations:
                        x,y,z = combination
                        good_combination = combination
                        vector = np.array([grid*x,grid*y,grid*z])
                        # Check overlap
                        for i in range(index):
                            atom_indices = atom_indices_for_each_molecule[i]
                            center,radius = geometry_info[tuple(atom_indices)]
                            if np.linalg.norm(vector + current_center - center) < radius + current_radius:
                                good_combination = None
                                break
                        if good_combination is not None:
                            overlap = False
                            break
                    if good_combination is not None:
                        break
                    l += 1
                # Move current center
                x,y,z = good_combination
                vector = np.array([grid*x,grid*y,grid*z])
                for atom_index in list(current_atom_indices):
                    coordinate_list[atom_index] += vector
                geometry_info[current_atom_indices][0] += vector
                index += 1
        return coordinates
        
    def visualize(self,method='rdkit'):
        # If it already has optimized geometry, use those geometires
        molecule_list = self.get_molecule_list()
        
        for molecule in molecule_list:
            coordinate_list = molecule.get_coordinate_list()
            count_zeros = 0
            for coordinate in coordinate_list:
                d = np.linalg.norm(np.array(coordinate))
                if d < 0.001:
                    count_zeros += 1
            if count_zeros > 1:
                coordinate_list = molecule.make_3d_coordinate(method)
                process.locate_molecule(molecule,coordinate_list)
        process.scatter_molecules(molecule_list)
        self.print_coordinate_list()
        self.write_geometry('tmp.xyz')
        import os
        os.system('molden tmp.xyz')  
        os.system('rm tmp.xyz')
