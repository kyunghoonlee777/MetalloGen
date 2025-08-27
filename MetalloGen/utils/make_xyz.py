import chem

import sys

molecule = chem.Intermediate(sys.argv[1])
chg = molecule.get_chg()
multiplicity = molecule.get_multiplicity()
try:
    library = sys.argv[2]
except:    
    library = 'rdkit'
molecule = molecule.sample_conformers(1,library)[0]

try:
    decimal = int(sys.argv[3])
except:
    decimal = 4

print (chg, multiplicity)
molecule.print_coordinate_list()
