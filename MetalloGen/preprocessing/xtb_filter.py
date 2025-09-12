import process
import sys, os, shutil
import pickle
from collections import defaultdict
import itertools
import numpy as np

from scipy.spatial.distance import cdist

from MetalloGen import chem
from MetalloGen import globalvars as gv
from MetalloGen.utils import shape

def main(args):
    # Arguments ...
    data_directory = args.data_directory
    geometriy_name = args.geometry_name
    calculator_name = args.calculator_name
    
    xtb_relax_dir = os.path.join(data_directory, "xtb_relaxed_geometries", calculator_name)
    result_dir = os.path.join(xtb_relax_dir, f"result/{geometry_name}")
    statistics_dir = os.path.join(xtb_relax_dir, "statistics")
    analysis_dir = os.path.join(xtb_relax_dir, "analysis")
    success_dir = os.path.join(xtb_relax_dir, "success")
    fail_dir = os.path.join(xtb_relax_dir, "fail")

    # make directories ...
    if not os.path.exists(statistics_dir):
        os.makedirs(statistics_dir)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    if not os.path.exists(success_dir):
        os.makedirs(success_dir)
    if not os.path.exists(fail_dir):
        os.makedirs(fail_dir)
        
    # clear existing files ...
    statistics_dir = os.path.join(statistics_dir, f"{geometry_name}.txt")
    analysis_dir = os.path.join(analysis_dir, f"{geometry_name}.txt")
    success_dir = os.path.join(success_dir, geometry_name)
    fail_dir = os.path.join(fail_dir, geometry_name)
    if os.path.exists(success_dir):
        shutil.rmtree(success_dir)
    if os.path.exists(fail_dir):
        shutil.rmtree(fail_dir)
    os.makedirs(success_dir)
    os.makedirs(fail_dir)

    relaxed_om_directories = os.listdir(result_dir)

    rmsd_content = dict()
    metal_z_list = list(range(3,5)) + list(range(11,14)) + list(range(19,32)) + list(range(37,51)) + list(range(55,84)) + list(range(87,113))

    analysis = {
        "Total": 0, 
        "Success": 0, 
        "QC_failed": 0, 
        "Energy_None": 0, 
        "Hessian_None": 0, 
        "Negative_Frequency": 0, 
        "Shape_Change": 0, 
        "Adj_Change": 0,
        "Skip": 0,
    }

    for relaxed_om in relaxed_om_directories:
        print(relaxed_om)
        analysis["Total"] += 1
        
        om_success_dir = os.path.join(success_dir, f"{relaxed_om}.xyz")
        om_fail_dir = os.path.join(fail_dir, f"{relaxed_om}.xyz")
        files = os.listdir(os.path.join(result_dir, relaxed_om))
        exit_flag = False
        if len(files) == 0:
            print("No files found.")
            analysis["Skip"] += 1
            continue
        for file_name in files:
            if file_name.endswith(".xyz"):
                with open(os.path.join(result_dir, relaxed_om, file_name)) as f:
                    try:
                        molecule, info = process.read_molecule(f, "xyz")
                    except:
                        print("Error reading molecule.")
                        analysis["Skip"] += 1
                        exit_flag = True
                        break
                with open(os.path.join(result_dir, relaxed_om, file_name)) as f:
                    xyz_lines = f.readlines()
            else:
                with open(os.path.join(result_dir, relaxed_om, file_name)) as f:
                    info_lines = f.readlines()
                original_rmsd, pickle_dir = info_lines[5].strip(), info_lines[6].strip()
        if exit_flag:
            continue
        
        # Check if the calculation is successful ...
        if info_lines[0].strip() == "False":
            xyz_lines[1] = xyz_lines[1].strip()
            if info_lines[1].strip() == "True": 
                xyz_lines[1] += "\tQC_failed"
                analysis["QC_failed"] += 1
            elif info_lines[2].strip() == "True": 
                xyz_lines[1] += "\tEnergy_None"
                analysis["Energy_None"] += 1
            elif info_lines[3].strip() == "True": 
                xyz_lines[1] += "\tHessian_None"
                analysis["Hessian_None"] += 1
            elif info_lines[4].strip() == "True": 
                xyz_lines[1] += "\tNegative_Frequency"
                analysis["Negative_Frequency"] += 1
            else:
                xyz_lines[1] += "\tUnknown_Error"
                analysis["Skip"] += 1
            xyz_lines[1] += "\n"
            with open(om_fail_dir, "w") as f:
                f.writelines(xyz_lines) 
            continue
        
        # Check if the shape and adjacency matrix is correct ...
        z_list = molecule.get_z_list()
        radius_list = [chem.radius_dict[chem.periodic_table[i]] for i in z_list]
        n = len(radius_list)
        R = np.repeat(np.array(radius_list),n).reshape((n,n))
        R = R + R.T
        
        metal_index = None
        for i, z in enumerate(z_list):
            if z in metal_z_list:
                metal_index = i
                break
        if metal_index is None:
            print("Something went wrong ... No metal atom found.")
            analysis["Skip"] += 1
            continue
        with open(pickle_dir, "rb") as f:
            _, original_adj_matrix, _, _, _ = pickle.load(f)
        geometry_name = pickle_dir.split("/")[-4]
        coords = molecule.get_coordinate_list()
        original_binding_indices = shape.get_binding_sites(original_adj_matrix, metal_index)
        binding_vectors = []
        for index in original_binding_indices:
            binding_vectors.append(np.mean(coords[index], axis=0) - coords[metal_index])
        original_binding_indices = [index for indices in original_binding_indices for index in indices]
        binding_vectors = np.array(binding_vectors)
        binding_vectors = binding_vectors / np.linalg.norm(binding_vectors, axis=1)[:, np.newaxis]
        CN = len(binding_vectors)
        candidate_geometries = gv.CN_known_geometries_dict[CN]
        candidate_directions = {key: gv.known_geometries_vector_dict[key] for key in candidate_geometries}
        min_rmsd = 1e6
        best_assigned_indices = None
        most_similar_geometry = None
        for key in candidate_geometries:
            direction_vectors = candidate_directions[key]
            rmsd, assigned_indices = shape.shape_measure(binding_vectors, direction_vectors)
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                best_assigned_indices = assigned_indices
                most_similar_geometry = key
        if best_assigned_indices is None or most_similar_geometry != geometry_name:
            xyz_lines[1] = xyz_lines[1].strip()
            xyz_lines[1] += f"\tShape_Change\t{geometry_name}\t{most_similar_geometry}\n"
            analysis["Shape_Change"] += 1
            with open(om_fail_dir, "w") as f:
                f.writelines(xyz_lines)
            continue
        
        distance_matrix = cdist(coords, coords)
        np.fill_diagonal(distance_matrix, 1e6)
        adj_correct = False
        ratio_list = np.linspace(1.00, 1.39, 20)
        for ratio in ratio_list:
            adj_matrix = np.where(distance_matrix/R < ratio, 1, 0)
            binding_indices = np.where(adj_matrix[metal_index] == 1)[0]
            if set(binding_indices) == set(original_binding_indices):
                adj_correct = True
                break
        if not adj_correct:
            xyz_lines[1] = xyz_lines[1].strip()
            xyz_lines[1] += f"\tAdj_Change\n"
            analysis["Adj_Change"] += 1
            with open(om_fail_dir, "w") as f:
                f.writelines(xyz_lines)
            continue
        
        # Save successful relaxed geometry ...
        analysis["Success"] += 1
        with open(om_success_dir, "w") as f:
            f.writelines(xyz_lines)
            
        # Save analysis ...
        dRMSD = float(min_rmsd) - float(original_rmsd)
        adRMSD = abs(dRMSD)
        file_dir = pickle_dir.replace(".pkl", ".xyz")
        file_dir = file_dir.replace("om_pkl", "om_xyz")
        rmsd_content[relaxed_om] = (original_rmsd, min_rmsd, adRMSD, len(z_list), file_dir, om_success_dir)

    rmsd_content = sorted(rmsd_content.items(), key=lambda x: x[1][2])    
    interval_data = defaultdict(list)
    for key, value in rmsd_content:
        num_atom = value[3]
        interval = (num_atom // 10) * 10
        interval_key = f"{interval}-{interval+10}"
        interval_data[interval_key].append((key, value))
    interval_data = dict(interval_data)
    interval_data = dict(sorted(interval_data.items(), key=lambda x: int(x[0].split("-")[0])))

    mixed_content = []
    keys = list(interval_data.keys())
    for items in itertools.zip_longest(*[interval_data[key] for key in keys]):
        for item in items:
            if item is not None:
                mixed_content.append(item)

    with open(os.path.join(statistics_dir), "w") as f:
        for key, value in mixed_content:
            f.write(f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\t{value[4]}\t{value[5]}\n")
        f.write("\n")

    with open(analysis_dir, "w") as f:
        
        f.write("Error analysis:\n")
        for key, value in analysis.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Atom count analysis:\n")
        for key, value in interval_data.items():
            f.write(f"{key}: {len(value)}\n")
        f.write("\n")
        
        f.write("Metal count analysis:\n")
        metal_num_dict = defaultdict(int)
        for key, value in mixed_content:
            file_dir = value[4]
            metal_name = file_dir.split("/")[-2]
            if metal_name in metal_num_dict:
                metal_num_dict[metal_name] += 1
            else:
                metal_num_dict[metal_name] = 1
        for key, value in metal_num_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract organometallic complexes from raw data.")
    parser.add_argument("--data_directory", "-dd", type=str, help="Directory containing processed data files.", required=True)
    parser.add_argument("--geometry_name", "-gn", type=str, help="Name of the geometry set to process.", required=True)
    parser.add_argument("--calculator_name", "-cn", type=str, help="Quantum chemistry calculator to use (e.g., 'orca').", required=True)
    
    args = parser.parse_args()
    main(args)