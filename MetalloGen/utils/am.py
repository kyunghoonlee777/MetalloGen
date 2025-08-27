'''
--am.py--
It contains the information about atoms.
'''

def getElectronCount_Neutral(Type):
    """ It returns the number of electrons of transition metals used in electron counting (neutral method)
    Type - element
    returns: the number of electrons 
    """
    Element=str.upper(Type)
    NumElectrons=dict(SC=3,TI=4, V=5,CR=6,MN=7,FE=8,CO=9,NI=10,CU=11,ZN=12,\
                       Y=3,ZR=4,NB=5,MO=6,TC=7,RU=8,RH=9,PD=10,AG=11,CD=12,\
                      LU=3,HF=4,TA=5, W=6,RE=7,OS=8,IR=9,PT=10,AU=11,HG=12)
    return NumElectrons[Element]

def getTypefromZ(Z):
    """ It returns the element of the given atomic number.
    Z - atomic number
    returns: the element
    """
    Z=int(Z)-1
    periodic_table=[
        'H' ,                                                                                'He',
        'Li','Be',                                                  'B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
        'Na','Mg',                                                  'Al','Si','P' ,'S' ,'Cl','Ar',
        'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
        'Rb','Sr','Y' ,'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I' ,'Xe',
        'Cs','Ba',
                  'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
                  'Lu','Hf','Ta','W' ,'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
        'Fr','Ra',
                  'Ac','Th','Pa','U' ,'Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No',
                  'Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og',
    ]
    return periodic_table[Z]


def getEN(atom):
    """ It returns the electronegativity of the given element.
    atom - element
    returns: electronegativity
    """
    a=str.lower(atom)
    if a=='n': return 3.0
    elif a=='c': return 2.5
    elif a=='s': return 2.5
    elif a=='o': return 3.5
    elif a=='f': return 4.0
    elif a=='br': return 2.8
    else: return 0


def getMass(atom):
    """ It returns atomic mass of the given element.
    atom - element
    returns: atomic mass
    """
    a=str.lower(atom)

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
    
def getR(atom):
    """ It returns covalent bond radius of the given element.
    atom - element
    returns: covalent bond radius
    """
    a=str.lower(atom)

#reference : Dalton Trans., 2008, 2832-2838
    if a=='h': return 0.31
    elif a=='li': return 1.28
    elif a=='be': return 0.96
    elif a=='b': return 0.84
    elif a=='c': return 0.76
    elif a=='n': return 0.71
    elif a=='o': return 0.66
    elif a=='f': return 0.57
    elif a=='na': return 1.66
    elif a=='mg': return 1.41
    elif a=='al': return 1.21
    elif a=='si': return 1.11
    elif a=='p': return 1.07
    elif a=='s': return 1.05
    elif a=='cl': return 1.02
    elif a=='ar': return 0.76
    elif a=='k': return 2.03
    elif a=='ca': return 1.76
    elif a=='co': return 1.38 #1/2*(lowspin+highspin)
    #elif a=='co': return 1.26 #lowspin
    #elif a=='co': return 1.50 #highspin
    elif a=='fe': return 1.42 #1/2*(lowspin+highspin)
    elif a=='ni': return 1.24
    #elif a=='cr': return 1.39
    elif a=='ti': return 1.60
    elif a=='br': return 1.20
    elif a=='rh': return 1.42
    elif a=='pd': return 1.39
    elif a=='i': return 1.39
    elif a=='hf': return 1.75
    else: return 0

#reference : J. Chem. Phys. 41, 3199 (1964)
'''
    if a=='h': return 0.25
    elif a=='li': return 1.45
    elif a=='be': return 1.05
    elif a=='b': return 0.85
    elif a=='c': return 0.70
    elif a=='n': return 0.65
    elif a=='o': return 0.60
    elif a=='f': return 0.50
    elif a=='na': return 1.80
    elif a=='mg': return 1.50
    elif a=='al': return 1.25
    elif a=='si': return 1.10
    elif a=='p': return 1.00
    elif a=='s': return 1.00
    elif a=='cl': return 1.00
    elif a=='ar': return 0.71
    elif a=='k': return 2.20
    elif a=='ca': return 1.80
    elif a=='co': return 1.35
    else: return 0
'''

def getZ(atom):
    """ It returns an atomic number of the given element.
    atom - Element
    returns: atomic number
    """
    a=str.lower(atom)
    if a=='h': return 1
    elif a=='li': return 3
    elif a=='be': return 4
    elif a=='b': return 5
    elif a=='c': return 6
    elif a=='n': return 7
    elif a=='o': return 8
    elif a=='f': return 9
    elif a=='na': return 11
    elif a=='mg': return 12
    elif a=='al': return 13
    elif a=='si': return 14
    elif a=='p': return 15
    elif a=='s': return 16
    elif a=='cl': return 17
    elif a=='ar': return 18
    elif a=='k': return 19
    elif a=='ca': return 20
    elif a=='co': return 27
    elif a=='ni': return 28
    elif a=='ti': return 22
    elif a=='fe': return 26
    elif a=='br': return 35
    elif a=='rh': return 45
    elif a=='bb': return 1.5
    elif a=='lg': return 2.5
    elif a=='pd': return 46
    elif a=='hf': return 72
    elif a=='i': return 53
    else: return 0

def getVE(atom): 
    """ It returns the number of valence electrons of the given element.
    atom - element
    returns: the number of valence electrons
    """
    a=str.lower(atom)
    if a=='h': return 1
    elif a=='li': return 1
    elif a=='be': return 2
    elif a=='b': return 3
    elif a=='c': return 4
    elif a=='n': return 5
    elif a=='o': return 6
    elif a=='f': return 7
    elif a=='na': return 1
    elif a=='mg': return 2
    elif a=='al': return 3
    elif a=='si': return 4
    elif a=='p': return 5
    elif a=='s': return 6
    elif a=='cl': return 7
    elif a=='br': return 7
    elif a=='ar': return 8
    elif a=='k': return 1
    elif a=='ca': return 2
    elif a=='pd': return 10
    elif a=='i': return 7
    else: return 0

def MaxV(atom): 
    """ It returns the maximum valence (maximum number of bonds that can be formed with other atoms) of the given element.
    atom - element
    returns: maximum valence
    """
    a=str.lower(atom)
    if a=='c':   return 4
    elif a=='h': return 1
    elif a=='b': return 3
    #elif a=='b': return 4
    elif a=='be': return 2
    elif a=='o': return 2
    elif a=='n': return 4
    elif a=='li': return 1
    #elif a=='p': return 4
    #elif a=='s': return 4
    elif a=='si': return 4
    elif a=='p': return 5 #valence shell expansion
    elif a=='s': return 6 #valence shell expansion
    elif a=='f': return 1
    elif a=='na': return 1
    elif a=='co':return 6
    elif a=='rh':return 6
    elif a=='ni':return 6 
    elif a=='ti':return 6
    elif a=='fe':return 6
    elif a=='cl': return 1
    elif a=='br': return 1
    elif a=='bb': return 3
    elif a=='lg': return 2
    elif a=='pd': return 6
    elif a=='i': return 3
    elif a=='al': return 3
    else: return 6

def getBL(a1,a2,coeff=1.10):
    """ Given two elements, it returns the upper limit of interatomic distance that can be regarded as a covalent bond.
    a1, a2 - element of two atoms
    return: upper limit of covalent bond length between two atoms 
    """
    atom1=str.lower(a1)
    atom2=str.lower(a2)
    return coeff*(getR(atom1)+getR(atom2))


