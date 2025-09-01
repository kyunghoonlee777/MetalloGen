from MetalloGen import chem, process

import numpy as np
import pickle

def is_organometallic(molecule,metal_z_list):
    for atom in molecule.atom_list:
        atomic_number = atom.get_atomic_number()
        if atomic_number in metal_z_list:
            return True
    return False


def extract_organometallics(intermediate,metal_z_list):
    molecule_list = intermediate.get_molecule_list()
    oms = []
    for molecule in molecule_list:
        if is_organometallic(molecule,metal_z_list):
            oms.append(molecule)
    return oms

def get_adj_from_sdf(intermediate,sdf_directory):
    
    n = len(intermediate.atom_list)
    adj_matrix = np.zeros((n,n))
    chg_info = dict()

    with open(sdf_directory,'r') as f:
        lines = f.readlines()
        
        for line in lines:
            if 'CHG' in line:
                infos = line.strip().split()
                m = int(infos[2])
                for k in range(m):
                    index = int(infos[2*k+3]) - 1
                    chg = int(infos[2*k+4])
                    chg_info[index] = chg
            elif len(line) != 10:
                continue
            
            parts = line[0:3], line[3:6], line[6:9], line[9:10]
            parts = [part.strip() for part in parts]
            
            if parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                i = int(parts[0]) - 1
                j = int(parts[1]) - 1
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1

    return adj_matrix, chg_info



def main(args):
    import os

    xyz_directory = args.xyz_directory
    sdf_directory = args.sdf_directory
    save_directory = args.save_directory
    start = args.start
    end = args.end
    num_interval = args.interval
   
    metal_z_list = list(range(11,14)) + list(range(19,32)) + list(range(37,51)) + list(range(55,84)) + list(range(87,113))
    
    save_directory = os.path.join(save_directory,'om/om_raw/')
    os.makedirs(save_directory,exist_ok=True)

    file_names = os.listdir(xyz_directory)
    om_info_list = []
    om_ids = set([])
    file_start = 0

    end = min(end, len(file_names))

    print (f"Loading data from {start} to {end} ...")
    index = 0
    print ("Processing data ...")

    for index,file_name in enumerate(file_names[start:end]):
        sdf_file_name = file_name.replace('.xyz','.sdf')
        if 'struct' in file_name:
            continue
        try:
            intermediate = chem.Intermediate(os.path.join(xyz_directory,file_name))
            total_adj, chg_info = get_adj_from_sdf(intermediate, os.path.join(sdf_directory,sdf_file_name))
            intermediate.adj_matrix = total_adj
            n = len(total_adj)
            intermediate.atom_feature['chg'] = np.zeros((n))
            for idx in chg_info:
                intermediate.atom_feature['chg'][idx] = chg_info[idx]
        except:
            continue
        oms = extract_organometallics(intermediate,metal_z_list)
        if len(oms) == 0:
            continue
        cnt = 0
        for i in range(len(oms)):
            if len(oms) > 1:
                om_name = file_name.split('.')[0] + f'_{cnt+1}'
            else:
                om_name = file_name.split('.')[0]
            om = oms[i]
            om_id = om.get_connectivity_id()
            if om_id in om_ids:
                continue

            om_ids.add(om_id)
            cnt += 1

            # Only save basic information
            z_list = om.get_z_list()
            adj_matrix = om.get_adj_matrix()
            chg_list = om.get_chg_list()
            coords = om.get_coordinate_list()
            chg = np.sum(chg_list)
            om_info_list.append([om_name, z_list, adj_matrix, chg_list, coords])

        if (index+1) % num_interval == 0 or index >= end -1:
            print (f'{index + 1} data was processed ...')
            file_end = min(end, index + 1)
            new_save_directory = os.path.join(save_directory,f'om_info_{file_start}_{file_end}.pkl')
            print (f'Saved at {new_save_directory}')

            with open(new_save_directory,'wb') as f:
                pickle.dump(om_info_list,f)
            om_info_list = []
            file_start = file_end



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract organometallic complexes from raw data.")
    parser.add_argument("--xyz_directory", "-xyz", type=str, help="Directory containing XYZ files of complexes.", required=True, )
    parser.add_argument("--sdf_directory", "-sdf", type=str, help="Directory containing SDF files of complexes.", required=True)
    parser.add_argument("--save_directory", "-sd", type=str, help="Directory to save extracted organometallic complexes.", defualt="./om/om_raw/")
    parser.add_argument("--start", "-s", type=int, help="Start index for processing files.", default=0)
    parser.add_argument("--end", "-e", type=int, help="End index for processing files.", default=100) 
    parser.add_argument("--interval", "-i", type=int, help="Number of files to process before saving intermediate results.", default=1000)
    
    args = parser.parse_args()
    main(args)
