import sys
import os

import numpy as np
import itertools
import string
import pickle

from MetalloGen.utils import am

def main(args):
    data_directory = args.data_directory
    om_save_directory = os.path.join(data_directory,'om/')
    if not os.path.exists(om_save_directory):
        print (f'Error: {om_save_directory} does not exist!')
        sys.exit(1)
    ligand_save_directory = os.path.join(data_directory,'ligand/')
    if not os.path.exists(ligand_save_directory):
        print (f'Error: {ligand_save_directory} does not exist!')
        sys.exit(1)

    # Make om folders ... 
    om_xyz_directory = os.path.join(om_save_directory,'om_xyz')
    om_pickle_directory = os.path.join(om_save_directory,'om_pkl')

    os.makedirs(om_xyz_directory,exist_ok = True)
    os.makedirs(om_pickle_directory,exist_ok = True)

    # Make ligand folders ...
    ligand_xyz_directory = os.path.join(ligand_save_directory,'ligand_xyz')
    ligand_pickle_directory = os.path.join(ligand_save_directory,'ligand_pkl')

    os.makedirs(om_xyz_directory,exist_ok = True)
    os.makedirs(om_pickle_directory,exist_ok = True)

    om_tmp_save_directory = os.path.join(om_save_directory,'om_tmp')
    ligand_tmp_save_directory = os.path.join(ligand_save_directory,'ligand_tmp')

    ligand_tmp_files = os.listdir(ligand_tmp_save_directory)
    om_tmp_files = os.listdir(om_tmp_save_directory)

    om_infos = dict() 
    for file_name in om_tmp_files:
        with open(f'{om_tmp_save_directory}/{file_name}','rb') as f:
            om_infos.update(pickle.load(f))

    ligand_infos = dict() 
    for file_name in ligand_tmp_files:
        with open(f'{ligand_tmp_save_directory}/{file_name}','rb') as f:
            ligand_infos.update(pickle.load(f))
    
    # Rename and save ligands ...
    alphabet = string.ascii_uppercase
    ligand_names = itertools.product(alphabet,repeat=5)

    ligand_dict = dict() # ligand_id -> ligand_name
    ligand_directories = {
        'haptic': dict(),
        'non_haptic': dict()
    } # is_haptic/denticity/name -> (ligand, binding_info)
    
    # Make ligand directory ...
    for ligand_id in ligand_infos:
        ligand_name = ''.join(next(ligand_names))
        ligand_dict[ligand_id] = ligand_name
        ligand_result = ligand_infos[ligand_id]
        is_haptic, denticity, hapticity_infos = ligand_result['result']
        if is_haptic:
            under_ligand_directories = ligand_directories['haptic']
            if denticity not in under_ligand_directories:
                under_ligand_directories[denticity] = dict()
            under_ligand_directories[denticity][ligand_name] = ligand_result
        else:
            under_ligand_directories = ligand_directories['non_haptic']
            if denticity not in under_ligand_directories:
                under_ligand_directories[denticity] = dict()
            under_ligand_directories[denticity][ligand_name] = ligand_result

                
    ligand_f1 = open(os.path.join(ligand_save_directory,'ligand_directory.txt'),'w')
    ligand_f2 = open(os.path.join(ligand_save_directory,'ligand_summary.txt'),'w')
    
    ligand_f1.write('Name\t\tDirectory\n')
    ligand_f2.write('Name\tIs haptic\tdenticity\tBinding type\n')

    print('Saving ligands and OMs ...')

    # Save ligand in pkl ...
    for is_haptic in ligand_directories:
        xyz_directory1 = os.path.join(ligand_xyz_directory,is_haptic)
        pickle_directory1 = os.path.join(ligand_pickle_directory,is_haptic)
        os.makedirs(xyz_directory1,exist_ok=True)
        os.makedirs(pickle_directory1,exist_ok=True)
        for denticity in ligand_directories[is_haptic]:
            xyz_directory2 = os.path.join(xyz_directory1,str(denticity))
            pickle_directory2 = os.path.join(pickle_directory1,str(denticity))
            os.makedirs(xyz_directory2,exist_ok=True)
            os.makedirs(pickle_directory2,exist_ok=True)
            
            for ligand_name in ligand_directories[is_haptic][denticity]:
                ligand_result = ligand_directories[is_haptic][denticity][ligand_name]
                xyz_file_directory = os.path.join(xyz_directory2,f'{ligand_name}.xyz')
                pickle_file_directory = os.path.join(pickle_directory2,f'{ligand_name}.pkl')
                check_haptic, denticity, hapticity_infos = ligand_result['result']
                z_list, adj_matrix, chg, coords, binding_info = ligand_result['pkl'] 
                hapticity_infos = [str(hapticity) for hapticity in hapticity_infos]

                # Save pickle file ...
                with open(pickle_file_directory,'wb') as f:
                    pickle.dump(ligand_result['pkl'],f)

                # Save xyz file ...
                with open(xyz_file_directory,'w') as f:
                    f.write(str(len(z_list))+'\n')
                    binding_content = []
                    for indices in binding_info:
                        binding_content.append(','.join([str(i) for i in indices]))
                    f.write(' '.join(binding_content)+'\n')
                    for i in range(len(z_list)):
                        element = am.getTypefromZ(z_list[i])
                        x, y, z = coords[i]
                        f.write(f'{element} {x} {y} {z}\n')
                        
                # Write summary and directory        
                ligand_f1.write(f'{ligand_name}\t{xyz_file_directory}\n')
                hapticity_content = '_'.join(hapticity_infos)
                ligand_f2.write(f'{ligand_name}\t{check_haptic}\t{denticity}\t{hapticity_content}\n')

    ligand_f1.close()
    ligand_f2.close()   
           
    om_directories = dict() # geometry/denticity/metal_name -> [(om,ligand_name,file_name)]
    
    # Make om directories ...
    for om_name in om_infos:
        om_result = om_infos[om_name]        
        #print (om_result['result'])

        metal_name, steric_number, geometry_info, ligand_ids, min_rmsd = om_result['result']  
        # Get denticity info
        binding_ligand_infos = []
        for ligand_id in ligand_ids:
            denticity = ligand_infos[ligand_id]['result'][1]
            binding_ligand_infos.append((ligand_id, denticity))
        binding_ligand_infos = sorted(binding_ligand_infos, key=lambda x:x[1],reverse=True)
        denticity_name = '_'.join([str(info[1]) for info in binding_ligand_infos])
        ligand_names = [ligand_dict[info[0]] for info in binding_ligand_infos]
        if type(geometry_info) is not str: # New geometry 
            continue
        if geometry_info not in om_directories:
            #om_directories[geometry_name] = {denticity_name: {metal_name: [(om,binding_ligands_names,file_name)]}}
            om_directories[geometry_info] = {denticity_name:{metal_name:dict()}}
            om_directories[geometry_info][denticity_name][metal_name][om_name] = om_result
        else:
            under_om_directories = om_directories[geometry_info]
            # Check denticity
            if denticity_name not in under_om_directories:
                under_om_directories[denticity_name] = {metal_name: dict()}
                under_om_directories[denticity_name][metal_name][om_name] = om_result
            else:
                under_om_directories = under_om_directories[denticity_name]
                if metal_name not in under_om_directories:
                    under_om_directories[metal_name] = dict()
                under_om_directories[metal_name][om_name] = om_result

    # Save OMs ...    
    om_f1 = open(os.path.join(om_save_directory,'om_directory.txt'),'w')
    om_f2 = open(os.path.join(om_save_directory,'om_summary.txt'),'w')
    
    om_f1.write('Name\tDirectory\n')
    om_f2.write('Name\tMetal\tGeometry name\tLigands\t\tminRMSD\n')        

    for geometry_name in om_directories:
        xyz_directory1 = os.path.join(om_xyz_directory,geometry_name)
        pickle_directory1 = os.path.join(om_pickle_directory,geometry_name)
        os.makedirs(xyz_directory1,exist_ok=True)
        os.makedirs(pickle_directory1,exist_ok=True)
        
        for denticity_name in om_directories[geometry_name]:
            xyz_directory2 = os.path.join(xyz_directory1,denticity_name)
            pickle_directory2 = os.path.join(pickle_directory1,denticity_name)
            os.makedirs(xyz_directory2,exist_ok=True)
            os.makedirs(pickle_directory2,exist_ok=True)
            
            for metal_name in om_directories[geometry_name][denticity_name]:
                xyz_directory3 = os.path.join(xyz_directory2,metal_name)
                pickle_directory3 = os.path.join(pickle_directory2,metal_name)
                os.makedirs(xyz_directory3,exist_ok=True)
                os.makedirs(pickle_directory3,exist_ok=True)
                
                for om_name in om_directories[geometry_name][denticity_name][metal_name]:
                    om_result = om_directories[geometry_name][denticity_name][metal_name][om_name]
                    xyz_file_directory = os.path.join(xyz_directory3,f'{om_name}.xyz')
                    pickle_file_directory = os.path.join(pickle_directory3,f'{om_name}.pkl')
                    z_list, adj_matrix, chg_list, coords, metal_index = om_result['pkl']
                    metal_name, steric_number, geometry_info, ligand_ids, min_rmsd = om_result['result']
                    chg = int(np.sum(chg_list))
                    # Save pickle file ...
                    with open(pickle_file_directory,'wb') as f:
                        pickle.dump(om_result['pkl'],f)

                    ligand_names = [ligand_dict[ligand_id] for ligand_id in ligand_ids]
                    ligand_content = '_'.join(ligand_names)
                    # Save xyz file ...
                    with open(xyz_file_directory,'w') as f:
                        f.write(str(len(z_list))+'\n')
                        content = ligand_content + f' {chg} {metal_index}'  # Ligand name, charge ...
                        f.write(content+'\n')
                        for i in range(len(z_list)):
                            element = am.getTypefromZ(z_list[i])
                            x, y, z = coords[i]
                            f.write(f'{element} {x} {y} {z}\n')
                    
                    # Write summary ...
                    om_f1.write(f'{om_name}\t{xyz_file_directory}\n')
                    om_f2.write(f'{om_name}\t{metal_name}\t{geometry_name}\t{ligand_content}\t{min_rmsd}\n')

    om_f1.close()
    om_f2.close()

    print('Done!')
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract organometallic complexes from raw data.")
    parser.add_argument("--data_directory", "-dd", type=str, help="Data directory containing processed data files.", required=True)
    
    args = parser.parse_args()
    main(args)