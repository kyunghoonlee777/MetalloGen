""" 
Geometry information vectors data module.
Many emulated from: https://en.wikipedia.org/wiki/Molecular_geometry 
Except where otherwise indicated.

Developed by Michael Taylor and Jan Janssen.
"""

import numpy as np

CN_known_geometry_dict = {
    2:  ['2_linear', '2_bent_135', '2_bent_90'],
    3:  ['3_trigonal_planar', '3_trigonal_pyramidal', '3_t_shaped'],
    4:  ['4_tetrahedral', '4_seesaw', '4_square_planar'],
    5:  ['5_trigonal_bipyramidal', '5_square_pyramidal', '5_pentagonal_planar'],
    6:  ['6_octahedral', '6_trigonal_prismatic', '6_pentagonal_pyramidal', '6_hexagonal_planar'],
    7:  ['7_pentagonal_bipyramidal', '7_hexagonal_pyramidal', '7_capped_trigonal_prismatic', '7_capped_octahedral'],
    8:  ['8_squre_antiprismatic', '8_sqaure_prismatic', '8_bicapped_trigonal_prismatic', '8_hexagonal_bipyramidal', '8_dodecahedral', '8_axial_bicapped_trigonal_prismatic'],
    9:  ['9_capped_square_antiprismatic', '9_tricapped_trigonal_prismatic', '9_tri_tri_mer_capped'],
    10: ['10_axial_bicapped_hexagonal_planar',],
    11: [],
    12: ['12_penta_bi_capped_pyramidal'],
}

known_geometry_vector_dict = {
    #### CN 2
    '2_linear': np.array([[0,0,1],[0,0,-1]]),
    '2_bent_135': np.array([[0,0,1],[0,1/np.sqrt(2),-1/np.sqrt(2)]]),
    '2_bent_90': np.array([[0,0,1],[0,1,0]]),
    #### CN 3
    '3_trigonal_planar': np.array([[2,0,0],[-1,np.sqrt(3),0],[-1,-np.sqrt(3),0]])/2,
    '3_trigonal_pyramidal': np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]]/np.sqrt(3)),
    '3_t_shaped': np.array([[0,0,1],[0,1,0],[0,0,-1]]),
    #### CN 4
    '4_tetrahedral': np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]/np.sqrt(3)),
    '4_square_planar': np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]),
    '4_seesaw': np.array([[0,0,1],[0,0,-1],[1,0,0],[0,1,0]]),
    #### CN 5
    '5_trigonal_bipyramidal': np.array([[0,0,2],[0,0,-2],[2,0,0],[-1,np.sqrt(3),0],[-1,-np.sqrt(3),0]])/2,
    '5_square_pyramidal': np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1]]),
    '5_pentagonal_planar': np.array([[1,0,0],[np.cos(2*np.pi/5),np.sin(2*np.pi/5),0],[np.cos(4*np.pi/5),np.sin(4*np.pi/5),0],[np.cos(6*np.pi/5),np.sin(6*np.pi/5),0],[np.cos(8*np.pi/5),np.sin(8*np.pi/5),0]]),
    #### CN 6
    '6_octahedral': np.array([[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]),
    '6_pentagonal_pyramidal': np.array([[0,0,1],[1,0,0],[np.cos(2*np.pi/5),np.sin(2*np.pi/5),0],[np.cos(4*np.pi/5),np.sin(4*np.pi/5),0],[np.cos(6*np.pi/5),np.sin(6*np.pi/5),0],[np.cos(8*np.pi/5),np.sin(8*np.pi/5),0]]),
    '6_trigonal_prismatic': np.array([[1.511, 0.0, 1.31],[-0.756, 1.309, 1.309],[-0.756, -1.309, 1.309],[1.511, 0.0, -1.31],[-0.756, 1.309, -1.309],[-0.756, -1.309, -1.309]])/2,
    '6_hexagonal_planar': np.array([[1,0,0],[np.cos(2*np.pi/6),np.sin(2*np.pi/6),0],[np.cos(4*np.pi/6),np.sin(4*np.pi/6),0],[np.cos(6*np.pi/6),np.sin(6*np.pi/6),0],[np.cos(8*np.pi/6),np.sin(8*np.pi/6),0],[np.cos(10*np.pi/6),np.sin(10*np.pi/6),0]]),
    #### CN 7
    '7_pentagonal_bipyramidal': np.array([[0,0,1],[0,0,-1],[1,0,0],[np.cos(2*np.pi/5),np.sin(2*np.pi/5),0],[np.cos(4*np.pi/5),np.sin(4*np.pi/5),0],[np.cos(6*np.pi/5),np.sin(6*np.pi/5),0],[np.cos(8*np.pi/5),np.sin(8*np.pi/5),0]]),
    '7_capped_trigonal_prismatic': np.array([[1.511, 0.0, 1.31],[-0.756, 1.309, 1.309],[-0.756, -1.309, 1.309],[1.511, 0.0, -1.31],[-0.756, 1.309, -1.309],[-0.756, -1.309, -1.309],[-2.0, 0.0, 0.0]])/2,
    '7_capped_octahedral': np.array([[0,0,2],[0,0,2],[0,2,0],[0,-2,0],[2,0,0],[-2,0,0],[1.155,1.155,1.155]])/2,
    '7_hexagonal_pyramidal': np.array([[0,0,1],[1,0,0],[np.cos(2*np.pi/6),np.sin(2*np.pi/6),0],[np.cos(4*np.pi/6),np.sin(4*np.pi/6),0],[np.cos(6*np.pi/6),np.sin(6*np.pi/6),0],[np.cos(8*np.pi/6),np.sin(8*np.pi/6),0],[np.cos(10*np.pi/6),np.sin(10*np.pi/6),0]]),
    #### CN 8
    '8_squre_antiprismatic': np.array([[-1, 1, 1],[1, 1, 1],[-1, -1, 1],[1, -1, 1],[-1.41, 0, -1],[0, -1.41, -1],[1.41, 0, -1],[0, 1.41, -1]])/np.sqrt(3),
    '8_sqaure_prismatic': np.array([[-1, 1, 1],[1, 1, 1],[-1, -1, 1],[1, -1, 1],[-1, 1, -1],[-1, -1, -1],[1, -1, -1],[1, 1, -1]])/np.sqrt(3),
    '8_bicapped_trigonal_prismatic': np.array([[1.511, 0.0, 1.31],[-0.756, 1.309, 1.309],[-0.756, -1.309, 1.309],[1.511, 0.0, -1.31],[-0.756, 1.309, -1.309],[-0.756, -1.309, -1.309],[-2.0, 0.0, 0.0],[1.16, -1.62, 0]])/2,
    '8_hexagonal_bipyramidal': np.array([[0,0,1],[0,0,-1],[1,0,0],[np.cos(2*np.pi/6),np.sin(2*np.pi/6),0],[np.cos(4*np.pi/6),np.sin(4*np.pi/6),0],[np.cos(6*np.pi/6),np.sin(6*np.pi/6),0],[np.cos(8*np.pi/6),np.sin(8*np.pi/6),0],[np.cos(10*np.pi/6),np.sin(10*np.pi/6),0]]),
    '8_dodecahedral': np.array([[1.289, 0.411, 0],[-1.289, 0.411, 0],[0, -0.411, 1.289],[0, -0.411, -1.289],[1, -1.568, 0],[-1, -1.568, 0],[0, 1.568, 1],[0, 1.568, -1]]),
    '8_axial_bicapped_trigonal_prismatic': np.array([[-0.77660752, 0.7455247, 1.68554848],[0.84666207, -0.75727505, -1.64611599],[-1.89364682, 0.64292567, -0.02735548],[-1.1421033, -0.44716888, -1.57975949],[-0.05187737, -1.12955356, 1.64967193],[1.08122157, -1.67863101, 0.11470761],[0.73916493, 1.67714792, 0.80050613],[1.25857889, 1.12625782, -1.07122477]])/2,
    #### CN 9
    '9_capped_square_antiprismatic': np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1], [-1.41, 0, -1], [0, -1.41, -1], [1.41, 0, -1], [0, 1.41, -1], [0, 0, 1.73]]),
    '9_tricapped_trigonal_prismatic': np.array([[1.511, 0.0, 1.31], [-0.756, 1.309, 1.309], [-0.756, -1.309, 1.309], [1.511, 0.0, -1.31], [-0.756, 1.309, -1.309], [-0.756, -1.309, -1.309], [-2.0, 0.0, 0.0], [1.16, -1.62, 0], [1.16, 1.62, 0]]),
    '9_tri_tri_mer_capped': np.array([
        [-0.32238184,  0.16659695, -1.96680335],
        [ 1.82278439, -0.47604810,  0.67144270],
        [-1.55384060,  0.32203597,  1.21732174],
        [-1.56677590, -0.94884186, -0.80306439],
        [ 1.26907643,  1.11747728, -1.06803068],
        [ 0.88550201, -1.62631150, -0.75564351],
        [ 0.83666247,  1.16758890,  1.39166521],
        [-0.41252724, -1.36401551,  1.40331143],
        [-0.99299587,  1.73068560, -0.13669874]
    ]),
    #### CN 10
    '10_axial_bicapped_hexagonal_planar': np.array([
        [-1.86281576,  0.06020958, -0.72546002], # 0,1 trans
        [ 1.86281576,  0.06020958,  0.72546002],
        [-0.65427417, -1.79359695, -0.59576447],
        [-1.23433283,  1.57281414, -0.05175061],
        [-0.00766799,  0.70251884,  1.87254065],
        [-1.23148994, -0.52774663,  1.48489596],
        [ 0.65420987, -1.79364224,  0.59569870],
        [ 1.23433412,  1.57281579,  0.05166980],
        [ 0.00758642,  0.70251895, -1.87254094],
        [ 1.23140247, -0.52774392, -1.48496947]
    ]),
    #### CN 12
    '12_penta_bi_capped_pyramidal': np.array([
        [ 1.94669217, -0.00263606,  0.45867487], #0,1 trans
        [-1.94669638, -0.00263596, -0.45865703],
        [ 0.80732904, -1.82239409, -0.16461956],
        [ 1.04605529, -0.25227919, -1.68585988],
        [-0.41002572, -0.89431687, -1.74128580],
        [-0.42780953,  1.32200287, -1.43850100],
        [ 0.97354769,  1.61530666, -0.66557439],
        [-0.80732904, -1.82239409,  0.16461956],
        [-1.04614067, -0.25228214,  1.68580646],
        [ 0.41002572, -0.89431687,  1.74128580],
        [ 0.42774741,  1.32205074,  1.43847627],
        [-0.97360413,  1.61527759,  0.66556241]
    ]),
}

metal_spin_dict = {
    # Lanthanides
    'La': 0, 'Ce': 1, 'Pr': 2, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 6, 'Gd': 7, 'Tb': 6, 'Dy': 5,
    'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 1, 'Lu': 0,
    # Actinides
    'Ac': 0, 'Th': 0, 'Pa': 0, 'U':  2, 'Np': 3, 'Pu': 4, 'Am': 6, 'Cm': 7, 'Bk': 6, 'Cf': 5,
    'Es': 4, 'Fm': 3, 'Md': 2, 'No': 0, 'Lr': 0,
    # First row transition metals
    'Sc': 0, 'Ti': 0, 'V':  0, 'Cr': 3, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 1, 'Zn': 0, 
    # Second row transition metals 
    'Y':  0, 'Zr': 0, 'Nb': 0, 'Mo': 0, 'Tc': 2, 'Ru': 4, 'Rh': 2, 'Pd': 0, 'Ag': 0, 'Cd': 0, 
    # Third row transition metals
    'Hf': 0, 'Ta': 0, 'W':  0, 'Re': 1, 'Os': 0, 'Ir': 0, 'Pt': 0, 'Au': 2, 'Hg': 0,
    # 4th row transition metals
    'Rf': 1, 'Db': 0, 'Sg': 0, 'Bh': 0, 'Hs': 0,
    # Post-transition metals
    'Al': 0,
    'Ga': 0,
    'In': 0, 'Sn': 0,
    'Tl': 0, 'Pb': 0, 'Bi': 0,
    'Nh': 0, 'Fl': 0, 'Mc': 2, 'Lv': 2,
    # Alkali/earth metals
    'Li': 0, 'Be': 0, 
    'Na': 0, 'Mg': 0, 
    'K':  0, 'Ca': 0, 
    'Rb': 0, 'Sr': 0, 
    'Cs': 0, 'Ba': 0, 
    'Fr': 0, 'Ra': 0
}