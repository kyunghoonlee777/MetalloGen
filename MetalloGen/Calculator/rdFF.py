import os
from copy import deepcopy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import (
    UFFGetMoleculeForceField,
    UFFOptimizeMolecule,
)
import rdkit.ForceField.rdForceField as FF

### ace-reaction libraries ###
from MetalloGen import chem, process


def get_rd_mol3D(ace_mol):
    pos = ace_mol.get_coordinate_list()
    rd_mol = ace_mol.get_rd_mol()
    try:
        Chem.SanitizeMol(rd_mol)
        AllChem.EmbedMolecule(rd_mol)
    except:
        print ('Embedding failed ...')
        return None
    for i in range(rd_mol.GetNumAtoms()):
        xyz = (float(pos[i][0]), float(pos[i][1]), float(pos[i][2]))
        rd_mol.GetConformer().SetAtomPosition(i, xyz)
        #rd_mol.GetConformer().SetAtomPosition(i, pos[i])
    return rd_mol


class UFFOptimizer:
    """UFF Optimizer Class (utilizing RDKit)"""

    MAXITS = 200
    #ETOL = 0.1
    ETOL = 3.0

    def __init__(self):
        self.save_directory = ""  # rdkit does not need this
        self.working_directory = ""  # rdkit does not need this
        self.command = "rdkit.ForceField"  # rdkit does not need this
        self.opt = None

    ### Common functions
    def get_energy(self, molecule):
        rd_mol = get_rd_mol3D(molecule)
        opt = UFFGetMoleculeForceField(
            rd_mol,
            #vdwThresh=1.0,
            ignoreInterfragInteractions=False
        )
        opt.Initialize()
        return opt.CalcEnergy()

    def fixed_optimization(self,molecule,fixing_atoms):
        rd_mol = get_rd_mol3D(molecule)
        opt = UFFGetMoleculeForceField(rd_mol)
        for atom_index in fixing_atoms:
            opt.AddFixedPoint(atom_index)
        opt.Initialize()
        opt.Minimize()
        new_coords = np.array(opt.Positions()).reshape((-1, 3))
        #print (new_coords)
        process.locate_molecule(molecule, new_coords)
       
  
    def optimize_geometry(
        self, molecule, constraints=[], maximal_displacement = 0.0, k = 10,add_atom_constraint = False):
        
        if constraints is None:
            constraints = list()
        rd_mol = get_rd_mol3D(molecule)
        if rd_mol is None:
            print ('FF optimization did not work properly ...')
            return False
        try:
            opt = UFFGetMoleculeForceField(rd_mol)
        except:
            smiles = Chem.MolToSmiles(rd_mol)
            print ('Optimizing Molecule:',smiles)
            return False
        n = len(molecule.atom_list)
        #print ('Before opt')
        #molecule.print_coordinate_list()
        #print()
        # Add fix constraints
        for constraint in constraints:
            start, end = constraint
            d = float(molecule.get_distance_between_atoms(start,end))
            opt.UFFAddDistanceConstraint(start,end, True,-maximal_displacement, maximal_displacement, 5 * k)
        others = list(range(n))

        if add_atom_constraint:
            for atom_index in others:
                opt.UFFAddPositionConstraint(atom_index,maximal_displacement,k)

        is_overlapping = False
        for i in range(1):
            #print (f'{i+1}th uff optimized geometry')
            opt.Initialize()
            opt.Minimize()
            conformer = rd_mol.GetConformer()
            for j in range(n):
                element = molecule.atom_list[j].get_element()
                x, y, z = conformer.GetAtomPosition(j)
                #print (element, x, y, z)
                # check pairs of coordinates to prevent overlap
                for k in range(j):
                    xx, yy, zz = conformer.GetAtomPosition(k)
                    if (x-xx)**2 + (y-yy)**2 + (z-zz)**2 < 0.6**2:
                        print("Too close atoms are found ! Aborting UFF ...")
                        is_overlapping = True
                        break
                if is_overlapping:
                    break
        if is_overlapping:
            # Print optimized geometry ...
            print ('Geometry after UFF ...')
            for j in range(n):
                element = molecule.atom_list[j].get_element()
                x, y, z = conformer.GetAtomPosition(j)
                print (element, x, y, z)
            return False



        '''
        for i in range(max_cycles):
            energy = opt.CalcEnergy()
            #energy = 0
            #print("energy: ", energy)
            opt.Minimize(maxIts=1)
            if abs(energy - opt.CalcEnergy()) < e_tol:
                break
        '''
        new_coords = np.array(opt.Positions()).reshape((-1, 3))
        #print (new_coords)
        process.locate_molecule(molecule, new_coords)
        
        '''
        # One more time ...
        rd_mol = get_rd_mol3D(molecule)
        opt = UFFGetMoleculeForceField(
            rd_mol,
            #vdwThresh=1.0,
            ignoreInterfragInteractions=False
        )
        opt.Initialize()
        k = 10000
        for constraint in fixing_constraints:
            start,end,d = constraint
            opt.AddDistanceConstraint(start,end,d,d+0.0001,k)
        opt.Minimize(maxIts = max_cycles,energyTol=e_tol)
        new_coords = np.array(opt.Positions()).reshape((-1, 3))
        #print (new_coords)
        process.locate_molecule(molecule, new_coords)
        '''
        return True

    ### wrapper function for autoCG
    def relax_geometry(
        self,
        molecule,
        contraints=None,
        chg=None,
        multiplicity=None,
        file_name=None,
        num_relaxation=MAXITS,
        maximal_displacement=1000,
        save_directory=None,
    ):
        return self.optimize_geometry(
            molecule, max_cycles=max_cycles, e_tol=e_tol
        )

    def clean_geometry(self):
        pass

    ### dummy functions for compatibility with autoCG
    def change_working_directory(self, path):
        pass

    def clean_scratch(self):
        pass


if __name__ == "__main__":
    """
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    """

    mol = Chem.MolFromXYZFile("initial_ts.xyz")
    tsPos = mol.GetConformer().GetPositions()

    # mol = Chem.MolFromXYZFile("P.xyz")
    ace_mol = chem.Molecule("P.xyz")
    bo = process.get_bo_matrix_from_adj_matrix(ace_mol, chg=-1)
    ace_mol.bo_matrix = bo

    mol = ace_mol.get_rd_mol()
    Chem.SanitizeMol(mol)
    AllChem.EmbedMolecule(mol)

    for i in range(mol.GetNumAtoms()):
        mol.GetConformer(0).SetAtomPosition(i, tsPos[i])
    # Chem.SanitizeMol(mol)
    print(len(mol.GetBonds()))
    for bond in mol.GetBonds():
        print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # AllChem.EmbedMultipleConfs(mol, 1)

    for c in mol.GetConformers():
        before = c.GetPositions()

    Chem.MolToXYZFile(mol, "before.xyz")

    # UFF = UFFGetMoleculeForceField(mol)
    # UFF.Initialize()

    ffp = MMFFGetMoleculeProperties(mol)
    UFF = MMFFGetMoleculeForceField(mol, ffp)
    UFF.Initialize()

    MAXITS = 1000
    ETOL = 3.0
    # ETOL = 0.1
    # ETOL=1e-6
    XYZ = ""

    energy = UFF.CalcEnergy()
    print(f"Energy before opts(kcal/mol):", energy)
    XYZ += Chem.MolToXYZBlock(mol)

    for i in range(1, MAXITS + 1):
        UFF.Minimize(maxIts=3)
        print(f"Energy after {i} opts(kcal/mol):", UFF.CalcEnergy())
        XYZ += Chem.MolToXYZBlock(mol)
        if abs(energy - UFF.CalcEnergy()) < ETOL:
            print(f"***Convergence condition satisfied. (dE < {ETOL} kcal/mol)")
            break
        energy = UFF.CalcEnergy()
        # print(type((mol.GetConformer(-1).GetPositions())))

    with open("UFF.xyz", "w") as f:
        f.write(XYZ)
