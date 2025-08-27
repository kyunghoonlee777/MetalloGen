import numpy as np
import os

from scipy.spatial.transform import Rotation

import chem
import process

def sample_from_crest(molecule,working_directory = None,num_proc=1):
    if working_directory == None:
        working_directory = os.getcwd()
    current_directory = os.getcwd()
    os.chdir(working_directory)
    # Make input R.xyz
    molecule.write_geometry('R.xyz')

    # Next, run crest
    chg = molecule.get_chg()
    multiplicity = molecule.get_multiplicity()
    os.system(f'crest R.xyz -T {num_proc} --noreftopo --chrg {chg} --uhf {multiplicity-1}')

    # Finally, read conformers
    conformers = []
    with open('crest_conformers.xyz') as f:
        while True:
            try:
                conformer, info = process.read_molecule(f)
            except:
                break
            if len(conformer.atom_list) == 0:
                break
            conformers.append(conformer)
    #os.system('rm crest_conformers.xyz R.xyz')

    for conformer in conformers:
        conformer.bo_matrix = molecule.bo_matrix
        conformer.chg = molecule.get_chg()
        conformer.multiplicity = molecule.get_multiplicity()
    
    # Move to original directory ...
    os.chdir(current_directory)
   
    return conformers


if __name__ == '__main__':
    import sys
    molecule = chem.Molecule(sys.argv[1])
    conformer = molecule.sample_conformers(1)[0]
    try:
        working_directory = sys.argv[2]
    except:
        working_directory = None
    conformers = sample_from_crest(conformer,working_directory)
    for i,conformer in enumerate(conformers):
        print (f'{i+1}th conformer')
        conformer.bo_matrix = molecule.bo_matrix
        conformer.print_coordinate_list()

