import os
from copy import deepcopy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
import rdkit.ForceField.rdForceField as FF

### ace-reaction libraries ###
import chem
import process


def get_rd_mol_with_3D(ace_mol):
    pos = ace_mol.get_coordinate_list()
    rd_mol = ace_mol.get_rd_mol()
    Chem.SanitizeMol(rd_mol)
    AllChem.EmbedMolecule(rd_mol)
    for i in range(rd_mol.GetNumAtoms()):
        rd_mol.GetConformer().SetAtomPosition(i, pos[i])

    return rd_mol


class UFFOptimizer:
    """UFF Optimizer Class (utilizing RDKit)"""

    def __init__(self):
        self.save_directory = None  # rdkit does not need this
        self.working_directory = None  # rdkit does not need this
        self.command = "rdkit.ForceField"  # rdkit does not need this
        self.opt = None

    ### Common functions
    def get_energy(self, molecule):
        rd_mol = get_rd_mol_with_3D(molecule)
        opt = UFFGetMoleculeForceField(rd_mol,ignoreInterfragInteractions=False)
        opt.Initialize()
        return opt.CalcEnergy()

    def optimize_geometry(self, molecule, max_cycles: int = 100):
        
        rd_mol = get_rd_mol_with_3D(molecule)
        opt = UFFGetMoleculeForceField(rd_mol,ignoreInterfragInteractions = False)
        opt.Initialize()
        opt.Minimize(maxIts = max_cycles)
        # perform opt until all atoms are well separated
        num_atoms = rd_mol.GetNumAtoms()
        for i in range(5):
            # check separation between atoms
            new_coords = np.array(opt.Positions()).reshape((-1, 3))
            is_well_separated = True
            for atom_a in range(num_atoms):
                coords_a = new_coords[atom_a, :]
                for atom_b in range(atom_a):
                    coords_b = new_coords[atom_b, :]
                    if np.linalg.norm(coords_a - coords_b) < 0.6:
                        is_well_separated = False
                        break
                if not is_well_separated:
                    break
            if is_well_separated:
                # exit opt loop
                break
            opt.Initialize()
            opt.Minimize(maxIts = max_cycles)
        process.locate_molecule(molecule, new_coords)

    def relax_geometry(self,molecule,max_cycles: int = 100, energy_criteria: float = 3.0, microiteration: int = 5):
        rd_mol = get_rd_mol_with_3D(molecule)
        opt = UFFGetMoleculeForceField(rd_mol,ignoreInterfragInteractions = False)
        opt.Initialize()
        opt.Minimize(maxIts = max_cycles)
        num_iteration = int(max_cycles/microiteration) + 1

        # perform opt until all atoms are well separated
        num_atoms = rd_mol.GetNumAtoms()
        for i in range(num_iteration):
            # check separation between atoms
            new_coords = np.array(opt.Positions()).reshape((-1, 3))
            is_well_separated = True
            for atom_a in range(num_atoms):
                coords_a = new_coords[atom_a, :]
                for atom_b in range(atom_a):
                    coords_b = new_coords[atom_b, :]
                    if np.linalg.norm(coords_a - coords_b) < 0.6:
                        is_well_separated = False
                        break
                if not is_well_separated:
                    break
            if is_well_separated:
                # exit opt loop
                break
            energy = opt.CalcEnergy()
            opt.Initialize()
            opt.Minimize(maxIts = max_cycles)
            if abs(energy - opt.CalcEnergy()) < energy_criteria:
                break

        new_coords = np.array(opt.Positions()).reshape((-1, 3))
        process.locate_molecule(molecule, new_coords)


if __name__ == "__main__":
    mol = Chem.MolFromSmiles("CCCC")
    ace_mol = process.get_ace_mol_from_rd_mol(mol)
    uff_optimizer = UFFOptimizer()
    uff_optimizer.relax_geometry(ace_mol)


    exit()
