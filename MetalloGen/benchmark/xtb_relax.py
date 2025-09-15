import process
import sys, os, shutil
import cclib
import numpy as np
import pickle

from MetalloGen import globalvars as gv
from MetalloGen.utils import shape
from MetalloGen.Calculator import orca, xtb_gaussian

def write_output(content, new_atom_list, file_directory, om_info):
    n = len(new_atom_list)
    name = om_info['name']
    for i in range(n):
        atom = new_atom_list[i]
        element = atom.get_element()
        coord = atom.get_coordinate()
        content += f"{element}\t{coord[0]:.9f}\t{coord[1]:.9f}\t{coord[2]:.9f}\n"
    with open(os.path.join(file_directory, f"{name}.xyz"), 'w') as f:
        f.write(content)
    with open(os.path.join(file_directory, f"{name}.txt"), 'w') as f:
        for i in range(1,len(om_info)):
            key = list(om_info.keys())[i]
            f.write(f'{om_info[key]}\n')
        f.write('\n')
    
def main(args):
    # Arguments ...
    data_directory = args.data_directory
    geometry_name = args.geometry_name
    start = args.start
    end = args.end
    working_directory = args.working_directory
    calculator_name = args.calculator_name
    
    working_directory = os.path.join(os.getcwd(), working_directory)

    # Test settings ...
    atom_num_criteria = 150
    
    # Constants ...
    metal_z_list = list(range(3,5)) + list(range(11,14)) + list(range(19,32)) + list(range(37,51)) + list(range(55,84)) + list(range(87,113))
    tm_z_list = list(range(21,31)) + list(range(39,49)) + list(range(57,81)) + list(range(89,113))
    
    # Directories ...
    initial_statistics_directory = os.path.join(data_directory, 'statistics', f'{geometry_name}.txt')
    save_directory = os.path.join(data_directory, 'xtb_relaxed_geometries', calculator_name)

    result_directory = os.path.join(save_directory, 'result')
    os.makedirs(result_directory, exist_ok=True)

    result_directory = os.path.join(result_directory, geometry_name)
    os.makedirs(result_directory, exist_ok=True)

    log_save_directory = os.path.join(save_directory, f"logs/{geometry_name}/")
    os.makedirs(log_save_directory, exist_ok=True)

    # Read initial statistics ...
    om_infos = []
    with open(initial_statistics_directory) as f:
        lines = f.readlines()
        for line in lines:
            om_name, rmsd, file_directory = line.strip().split()
            rmsd = float(rmsd)
            om_infos.append((om_name, rmsd, file_directory))
            
    # Sort om_infos by rmsd ...
    om_infos = sorted(om_infos, key=lambda x:x[1]) 

    if end > len(om_infos):
        end = len(om_infos)

    # Initialize calculator ...
    if calculator_name == 'orca':
        G = orca.Orca()
    else:
        G = xtb_gaussian.XTB_Gaussian()

    # Relax structures ...
    cnt = 0
    for i in range(start, end):
        print(f'{i}th structure ...')
        om_name, original_rmsd, file_directory = om_infos[i]
        om_result_directory = os.path.join(result_directory, om_name)
        os.makedirs(om_result_directory, exist_ok=True)
    
        with open(file_directory) as f:
            molecule, info = process.read_molecule(f,'xyz')
        _, chg, metal_index = info # ligand names, charge, metal index
        chg = int(chg)
        metal_index = int(metal_index)
        
        pickle_directory = file_directory.replace('.xyz','.pkl')
        pickle_directory = pickle_directory.replace('om_xyz','om_pkl')
        with open(pickle_directory, 'rb') as f:
            _, adj_matrix, _ , _, _ = pickle.load(f)
            
        original_binding_indices = shape.get_binding_sites(adj_matrix, metal_index)
        for binding_indices in original_binding_indices:
            if len(binding_indices) > 1:
                cnt += 1
                break
            
        om_info = {'name':om_name, 
            'success':True, 
            'QC_fail':False, 
            'Energy_None':False, 
            'Hessian_None':False, 
            'Negative_freq':False,
            'original_rmsd':original_rmsd,
            'pickle_directory':pickle_directory}
        
        metal_atom = molecule.atom_list[metal_index]
        
        z_list = molecule.get_z_list()
        metal_z = metal_atom.get_atomic_number()
        n = len(z_list)
        
        # Set working directory ...
        om_working_directory = os.path.join(working_directory, f'{om_name}_{calculator_name}')
        os.makedirs(om_working_directory, exist_ok=True)
        G.change_working_directory(om_working_directory)

        index = z_list.tolist().index(metal_z)
        # Replace actinide with lanthanide ...
        if metal_z > 88 and metal_z < 104:
            metal_z -= 32
        metal_atom.set_atomic_number(metal_z)
        
        # Set charge and multiplicity ...
        element = metal_atom.get_element()
        element = element.lower().capitalize()
        mult = gv.metal_spin_dict[element] if element in gv.metal_spin_dict else 0
        z_sum = np.sum(z_list)
        e_sum = z_sum - chg
        if e_sum % 2 == 1:
            if mult == 0:
                mult = 1
            elif mult < 7 and mult % 2 == 0:
                mult += 1
            elif mult >= 7 and mult % 2 == 0:
                mult -= 1
        if e_sum % 2 == 0 and mult % 2 == 1:
            mult -= 1
        elif e_sum % 2 == 1 and mult % 2 == 0:
            mult += 1
        mult += 1
        
        molecule.chg = chg
        molecule.multiplicity = mult
        molecule.atom_list[index].set_atomic_number(metal_z)

        new_molecule = molecule.copy() # Optimizing molecule ...
        new_atom_list = new_molecule.get_atom_list()
        new_molecule.chg = chg
        new_molecule.multiplicity = mult
        
        content = f'{n}\n'
        content += f'{chg}\t{mult}\t'
        
        # Skip calculation if it contains too many atoms ...
        if n > atom_num_criteria:
            print (f'Skipped calculation because it contains more than {atom_num_criteria} atoms !!!')
            continue
        # Skip calculation if it contains nonmetal atom with valence larger than 4 ...
        high_valence_indices = np.where(np.sum(adj_matrix, axis=1) > 4)[0]
        filtered_indices = high_valence_indices[high_valence_indices != metal_index]
        if len(filtered_indices) > 0:
            print ('Skipped calculation because it contains nonmetal atom with valence larger than 5 !!!')
            continue
        # Skip calculation if it contains non-transition metal ...
        polynuclear = False
        for i, z in enumerate(z_list):
            if i == metal_index:
                continue
            if z in metal_z_list:
                polynuclear = True
        if polynuclear:
            print ('Skipped calculation because it contains more than one metal atom !!!')
            continue

        om_save_directory = os.path.join(result_directory, om_name)
        os.makedirs(om_save_directory, exist_ok=True)

        # Perform QC calculation ...
        try:
            G.optimize_geometry(new_molecule, file_name=om_name)
        except:
            print ('XTB relax failed ...')
            om_info['success'] = False
            om_info['QC_fail'] = True
            content += '\n'
            write_output(content, new_atom_list, om_result_directory, om_info)
            continue
    
        # om_info energy ...
        energy = new_molecule.energy
        if energy is not None:
            content += f'{energy}\t'
        else:
            print ('Energy is None ...')
            om_info['success'] = False
            om_info['Energy_None'] = True
            content += '\n'
            write_output(content, new_atom_list, om_result_directory, om_info)
            continue

        # om_info negative frequency ...
        force, hessian = G.get_hessian(new_molecule,file_name=f'{om_name}_freq')
        if force is None or hessian is None:
            om_info['success'] = False
            om_info['Hessian_None'] = True
            content += '\n'
            write_output(content, new_atom_list, om_result_directory, om_info)
            continue
        try:
            freq_directory = os.path.join(om_working_directory, f'{om_name}_freq.log')
            p = cclib.parser.Gaussian(freq_directory)
            data = p.parse()
            vibfreqs = data.vibfreqs
            min_freq = vibfreqs[0]
        except:
            om_info['success'] = False
            om_info['Hessian_None'] = True
            content += '\n'
            write_output(content, new_atom_list, om_result_directory, om_info)
            continue
        if min_freq < 0:
            om_info['success'] = False
            om_info['Negative_freq'] = True
            content += f'{min_freq}\n'
            write_output(content, new_atom_list, om_result_directory, om_info)
            continue
        # Save results ...
        content += f"{min_freq}\n"
        write_output(content, new_atom_list, om_result_directory, om_info)

        log_file = os.path.join(om_working_directory, f"{om_name}.log")
        shutil.copy(log_file, os.path.join(log_save_directory, f"{om_name}.log"))

        try:
            log_file = os.path.join(om_working_directory, f"{om_name}_freq.log")
            shutil.copy(log_file, os.path.join(log_save_directory, f"{om_name}_freq.log"))
        except:
            pass
    
    print(f'Number of complexes containing polyhapto ligands: {cnt}')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract organometallic complexes from raw data.")
    parser.add_argument("--data_directory", "-dd", type=str, help="Directory containing processed data files.", required=True)
    parser.add_argument("--geometry_name", "-gn", type=str, help="Name of the geometry set to process.", required=True)
    parser.add_argument("--start", "-s", type=int, help="Start index for processing files.", default=1)
    parser.add_argument("--end", "-e", type=int, help="End index for processing files.", default=1000) 
    parser.add_argument("--working_directory", "-wd", type=str, help="Working directory for calculations.", default=".")
    parser.add_argument("--calculator_name", "-cn", type=str, help="Quantum chemistry calculator to use (e.g., 'orca').", required=True)
    
    args = parser.parse_args()
    main(args)