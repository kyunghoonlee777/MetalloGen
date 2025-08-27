import numpy as np
import os
import math

def read_gradient(input_xtb_gradient):
    gradient = list()
    with open(input_xtb_gradient, 'r') as f:
        for line in f.readlines():
            line_sp = line.split()
            # skip lines without gradient data
            if len(line_sp) != 3:
                continue
            gradient.append(list(map(float, line_sp)))

    return np.array(gradient)
        


def read_hessian(input_xtb_hessian):
    with open(input_xtb_hessian, 'r') as f:
        input_xtb = f.read().replace("$hessian", "")

    # 1D to 2D
    hessian = np.array(list(map(float, input_xtb.split())))
    num_coords = int(np.sqrt(np.size(hessian)))
    return hessian.reshape((num_coords, num_coords))


def convert_gradient():
    pass


def convert_hessian(input_xtb_xyz, hessian_file, output_orca_hess):
    hessian = read_hessian(hessian_file)
    num_coords = hessian.shape[0]

    output_content = f"""
$orca_hessian_file

$hessian
{num_coords}
"""

    num_blocks = math.ceil(num_coords / 5)    

    for i in range(num_blocks):
        # display up to five
        remainder = num_coords - (i * 5)
        if  remainder > 5:
            remainder = 5

        # write topline
        output_content += ' '*2
        for j in range(remainder):
            current_coord = i*5+j
            output_content += f"{current_coord:>22}"
        output_content += "\n"

        # write other lines
        for vertical_idx in range(num_coords):
            output_content += f"{vertical_idx:>5}   "

            for j in range(remainder):
                horizontal_idx = i*5+j
                hessian_from_idx = hessian[horizontal_idx, vertical_idx]
                output_content += "  "
                output_content += f"{hessian_from_idx:>20E}"

            output_content += "\n"

    # $atoms is required
    num_atoms = num_coords // 3
    output_content += f"""
$atoms
{num_atoms}
"""
    
    # read input xyz and write structure
    with open(input_xtb_xyz, 'r') as f:
        lines = f.readlines()[2:]

    for line in lines:
        if len(line.strip()) == 0:
            break
        element_symbol, x, y, z = line.split()
        x, y, z = float(x), float(y), float(z)
        output_content += f"{element_symbol:>2}       0.000       {x:>.10f}  {y:>.10f}  {z:>.10f}\n"
    
    output_content += """

$end
"""

    with open(output_orca_hess, 'w') as f:
        f.write(output_content)

if __name__ == "__main__":
    pass

