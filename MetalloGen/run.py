"""
This is the main file for the MetalloGen package
MetalloGen is a package that generate 3D structures of organometallic complexes
"""

import os, shutil
import time
import argparse
import pickle
import glob
from pathlib import Path

import numpy as np

from MetalloGen import globalvars as gv
from MetalloGen import om, embed, clean_geometry

from MetalloGen.Calculator import orca, gaussian

from MetalloGen.utils import clustering


def clean_working_directory(working_directory):
    wd = Path(working_directory)
    wildcard_patterns = ["Gau-*"]
    file_names = ['energy','gradient','xtb','charges','xtbrestart','xtbtopo.mol']
    for pattern in wildcard_patterns:
        for file in wd.glob(pattern):
            if file.is_file():
                file.unlink()

    # Remove exact-named files
    for filename in file_names:
        file = wd / filename
        if file.exists() and file.is_file():
            file.unlink()


class TMCGenerator:

    def __init__(self,calculator,scale = 1.0,align = True,always_qc = False):
        self.cleaner = clean_geometry.TMCOptimizer(calculator = calculator)
        self.calculator = calculator
        self.scale = scale
        self.corrector = 0.2
        self.align = align
        self.always_qc = always_qc

    def sample_conformer(self,metal_complex,return_time = False):
        options = [0, 1]
        scale = self.scale
        align = self.align
        cleaner = self.cleaner
        
        metal_index = metal_complex.metal_index
        atom_list = metal_complex.get_atom_list()
        radius_list = [atom.get_radius() for atom in atom_list]
        n = len(radius_list)
        R = np.repeat(np.array(radius_list),n).reshape((n,n))
        R = R + R.T
        
        candidate_positions = []
        candidate_list = []
        energy_list = []
        
        gen_time_list = []
        scan_time_list = []
        gen_times = []
        scan_times = []
        
        for option in options:
            idx = options.index(option)
            print(f"Embedding molecule {idx+1}/{len(options)} ...")
            gen_st = time.time()
            positions = embed.get_embedding(metal_complex,scale,option,align,use_random = True)
            if positions is not None:
                gen_et = time.time()
                candidate_positions.append(positions)
                gen_time_list.append(gen_et - gen_st)

        # Clean geometry and compute energy for each candidate position
        for candidate_position in candidate_positions:
            scan_st = time.time()
            tmp_metal_complex = metal_complex.copy()
            tmp_metal_complex.set_position(candidate_position)
            success = cleaner.clean_geometry(tmp_metal_complex,scale,self.always_qc) # Geometry optimization
            scan_et = time.time()
            scan_time_list.append(scan_et - scan_st)
            ace_mol = tmp_metal_complex.get_molecule()
            energy = None

            # If QC calculation failed, it's a bad structure ...
            candidate_list.append(ace_mol)
            if energy is None:
                try:
                    energy = self.calculator.get_energy(ace_mol)
                except:
                    energy = None
                    scan_time_list[-1] = -1
            energy_list.append(energy if energy is not None else 1e6)

        # Use the most stable positions ...
        print ('energy',energy_list,'\n')
        if len(energy_list) == 0:
            if return_time:
                return [], [], []
            else:
                return []
        if min(energy_list) == 1e6:
            if return_time:
                return [], [], []
            else:
                return []
        indices = np.argsort(energy_list)
        ace_mols = []
        for i in indices:
            if energy_list[i] < 1e6:
                candidate_list[i].energy = energy_list[i]
                ace_mols.append(candidate_list[i])
                gen_times.append(gen_time_list[i])
                scan_times.append(scan_time_list[i])

        if return_time:
            return ace_mols, gen_times, scan_times
        else:
            return ace_mols

# Main function to run the experiment for generating 3D structure of CSD structures
def main():
    np.set_printoptions(threshold=1000)

    parser = argparse.ArgumentParser(
        description="Generate 3D structure of organometallic complex"
    )
    parser.add_argument("--smiles", "-s", type=str, help="Input MSMILES string")
    parser.add_argument("--input_directory", "-id", type=str, help="Input MOL/SDF file directory")
    parser.add_argument("--working_directory", "-wd", type=str, help="Scratch directory for running quantum chemical calculation", default=None)
    parser.add_argument("--save_directory", "-sd", type=str, help="Directory to save the results", default=None)
    parser.add_argument("--final_relax", "-r", type=int, help="Whether to perform final relaxation after generation", default=1)
    parser.add_argument("--num_conformer", "-nc", type=int, help="Number of conformers", default=10)
    parser.add_argument("--calculator", "-c", type=str, help="Calculator to use", default="xtb_gaussian")

    args = parser.parse_args()

    smiles = args.smiles
    input_directory = args.input_directory
    working_directory = args.working_directory
    save_directory = args.save_directory

    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
    if working_directory is not None:
        os.makedirs(working_directory, exist_ok=True)

    # Metal complex generation based on MSMILES
    if smiles and "|" in smiles:
        metal_complex = om.get_om_from_modified_smiles(smiles)
    elif input_directory:
        metal_complex = om.get_om_from_sdf(input_directory)
    else:
        raise Exception("Please provide either MSMILES string or MOL/SDF file directory ...")

    # Set up the calculator
    #calculator = orca.Orca()
    if args.calculator == "xtb_gaussian":
        calculator = gaussian.Gaussian()
        calculator.switch_to_xtb_gaussian()
    elif args.calculator == "orca":
        calculator = orca.Orca()
    elif args.calculator == "gaussian":
        calculator = gaussian.Gaussian()
    else:
        raise Exception("Invalid calculator ...")
    calculator.change_working_directory(working_directory)

    if smiles:
        print(f"MSMILES: {smiles}")
    else:
        file_name = os.path.basename(input_directory)
        print(f"Input file: {file_name}")
    print(f"num atoms: {len(metal_complex.get_atom_list())}")
    print(f"chg: {metal_complex.chg}")
    print(f"mult: {metal_complex.multiplicity}")

    if args.num_conformer < 10:
        scales = [0.70, 0.80, 0.90, 1.00, 1.10]
    else:
        scales = [0.7 + ((0.8 + 0.1*(args.num_conformer-10))/(args.num_conformer-2))*k for k in range(int(args.num_conformer/2))]
    energy_criteria = 100.0
    num_trial = 1
    
    neighbor_list = metal_complex.get_distances_from_center()

    num = 0
    initial_hessian = None
    #initial_hessian = 'calcfc'

    total_ace_mols = []
    for i, scale in enumerate(scales):
        print(f"\nGenerating conformer with scale {scale} ...")
        print("======================================================")
        generator = TMCGenerator(calculator,scale,True)
        ace_mols = generator.sample_conformer(metal_complex)
        total_ace_mols.extend(ace_mols)

    print("=======================================================")
    print(f"\nTotal {len(total_ace_mols)} conformers generated ...")

    print(f"\nClustering conformers ...")
    conformers = [ace_mol.get_coordinate_list() for ace_mol in total_ace_mols]
    atom_symbol = [atom.get_element() for atom in metal_complex.get_atom_list()]
    clusters, D = clustering.cluster_conformers_butina(conformers, atom_symbol, cutoff=0.5, exclude_H=True)

    dif_ace_mols = [total_ace_mols[i[0]] for i in clusters]
    print(f"{len(dif_ace_mols)} conformers after clustering ...")

    if not args.final_relax:
        exit()

    print("\nPerforming final relaxation ...")
    for i, ace_mol in enumerate(dif_ace_mols):
        print(f"\nRelaxing conformer {i+1}/{len(dif_ace_mols)} ...")
        print("======================================================")
        success = False
        for j in range(num_trial):
            original_energy = ace_mol.energy
            if original_energy == 1e6:
                break
            
            print("Generated geometry ...")
            print("Energy:", original_energy)
            ace_mol.print_coordinate_list()
            print("\nFinal relaxation ...")
            normal_termination = True
                
            if True:
                calculator.optimize_geometry(ace_mol, file_name=f"final_relax_{num}",initial_hessian=initial_hessian,save_directory=save_directory)
            else:
                normal_termination = False
        
            if not normal_termination:
                if True:
                    calculator.optimize_geometry(ace_mol, file_name=f"final_relax_{num+1}",initial_hessian=initial_hessian,save_directory=save_directory)
                else:
                    normal_termination = False

            # Reset working directory                 
            clean_working_directory(working_directory)
            #os.system(f"/bin/rm {working_directory}/Gau-*")
            #os.system(f"/bin/rm {working_directory}/energy")
            #os.system(f"/bin/rm {working_directory}/gradient")
            #os.system(f"/bin/rm {working_directory}/wbo")
            #os.system(f"/bin/rm {working_directory}/charges")
            #os.system(f"/bin/rm {working_directory}/xtbrestart")
            #os.system(f"/bin/rm {working_directory}/xtbtopo.mol")
            relaxed_energy = ace_mol.energy
            print ("Energy:", relaxed_energy)
            chg = ace_mol.chg
            mult = ace_mol.multiplicity
            content = f"{len(ace_mol.atom_list)}\n{chg}\t{mult}\t{relaxed_energy}\n"
            for atom in ace_mol.atom_list:
                content += atom.get_content()
                
            ace_mol.print_coordinate_list()
            print()
            
            if not normal_termination:
                print("Relaxation failed ...")
                continue
            else:
                # Here, we do not hessian calculation, because the geometry is simply for relaxing the structure 
                print ("Relaxation success!\n")
                print ("=======================================================")
                num += 1
                success = True

            if save_directory:
                conf_save_directory = os.path.join(save_directory,f"result_{num}.xyz")
                with open(conf_save_directory, "w") as f:
                    f.write(content)

            if success:
                break

        if num >= args.num_conformer:
            break

    print("=======================================================")
    print(f"\nTotal {num} conformers are successfully relaxed ...")
        
    
if __name__ == "__main__":
    main()
