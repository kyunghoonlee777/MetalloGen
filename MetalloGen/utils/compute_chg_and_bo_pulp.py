from typing import List, Dict, Tuple
from itertools import combinations as comb
import sys

from rdkit import Chem

import pulp as pl
import numpy as np
from scipy import spatial

periodic_table = ['H','He','Li','Be','B','C','N','O','F','Ne',\
    'Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn',\
    'Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',\
    'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',\
    'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',\
    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',\
    'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm']

EN_TABLE = {
    "H" : 2.300,
    "Li": 0.912,
    "Be": 1.576,
    "B" : 2.051,
    "C" : 2.544,
    "N" : 3.066,
    "O" : 3.610,
    "F" : 4.193,
    "Ne": 4.787,
    "Na": 0.869,
    "Mg": 1.293,
    "Al": 1.613,
    "Si": 1.916,
    "P" : 2.253,
    # "P" : 3.053, # phosphorus, adjusted
    "S": 2.589,
    # "S" : 3.089, # sulfur, adjusted
    "Cl": 2.869,
    # "Cl" : 3.369, # Halogen, adjusted
    "Ar": 3.242,
    "K" : 0.734,
    "Ca": 1.034,
    "Ga": 1.756,
    "Ge": 1.994,
    "As": 2.211,
    "Se": 2.424,
    "Br": 2.685,
    # "Br": 3.285, # Halogen, adjusted
    "Kr": 2.966,
    "Rb": 0.706,
    "Sr": 0.963,
    "In": 1.656,
    "Sn": 1.824,
    "Sb": 1.984,
    "Te": 2.158,
    "I" : 2.359 + 1,
    # "I": 3.159, # Halogen, adjusted
    "Xe": 2.582,
    "Fe": 1.83,
}  # J. Am. Chem. Soc. 1989, 111, 25, 9003–9014


def moSolve(prob, objs, verbose: bool):
    """multi-objective optimization. Simple modification of pulp.LpProblem.sequentialSolve"""
    statuses = []
    objvalues = []
    if not (prob.solver):
        prob.solver = pl.LpSolverDefault
        prob.solver.msg = False  # suppress the output
    for i, (_, obj, s) in enumerate(objs):
        prob.setObjective(obj)
        prob.sense = s
        status = prob.solver.actualSolve(prob)
        statuses.append(status)
        objvalues.append(obj.value())
        if verbose:
            prob.writeLP(f"record_obj{i}.lp")
        if s == pl.const.LpMinimize:
            prob += obj <= obj.value(), f"Obj_{i}"
        elif s == pl.const.LpMaximize:
            prob += obj >= obj.value(), f"Obj_{i}"
    return prob, statuses, objvalues


def get_adj_matrix_from_distance3(molecule):
    n = len(molecule.atom_list)
    radius_list = molecule.get_radius_list()
    radius_matrix = np.repeat(radius_list, n).reshape((n, n))
    criteria_matrix = (radius_matrix + radius_matrix.T) + np.ones(
        (n, n)
    ) * 0.45  # J. Chem. Inf. Comput. Sci. 1992, 32, 401-406
    coordinate_list = molecule.get_coordinate_list()
    distance_matrix = spatial.distance_matrix(coordinate_list, coordinate_list)
    adj = np.where(distance_matrix < criteria_matrix, 1, 0)
    np.fill_diagonal(adj, 0)
    return adj


def get_adj_matrix_from_distance4(molecule, coeff=1.15):
    n = len(molecule.atom_list)
    radius_list = molecule.get_radius_list()
    radius_matrix = np.repeat(radius_list, n).reshape((n, n))
    criteria_matrix = (radius_matrix + radius_matrix.T) * coeff
    coordinate_list = molecule.get_coordinate_list()
    distance_matrix = spatial.distance_matrix(coordinate_list, coordinate_list)
    adj = np.where(
        ((distance_matrix < criteria_matrix) | (distance_matrix < 0.80)), 1, 0
    )
    np.fill_diagonal(adj, 0)
    return adj


def get_ring_info(z_list, adj_matrix):
    chg_list = np.zeros(len(z_list))
    #print("z_list", z_list)
    new_z_list = adj_matrix.sum(axis=-1)
    #print("new_z_list", new_z_list)
    new_z_list[new_z_list == 1] = 1  # H
    new_z_list[new_z_list == 2] = 8  # O
    new_z_list[new_z_list == 3] = 7  # N
    new_z_list[new_z_list == 4] = 6  # C
    new_z_list[new_z_list == 5] = 15  # P
    new_z_list[new_z_list == 6] = 16  # S
    new_z_list[new_z_list >  6] = 26  # Fe

    new_rd = Chem.rdchem.RWMol()
    for z in new_z_list:
        new_rd.AddAtom(Chem.rdchem.Atom(int(z)))
    for b in np.vstack(np.triu(adj_matrix, k=1).nonzero()).T:
        new_rd.AddBond(int(b[0]), int(b[1]), Chem.rdchem.BondType.SINGLE)
    #print(Chem.MolToSmiles(new_rd))
    Chem.SanitizeMol(new_rd)

    sssrs = Chem.GetSymmSSSR(new_rd)
    RingInfo = new_rd.GetRingInfo()
    atoms_in_ring = RingInfo.AtomRings()
    bond_rings = RingInfo.BondRings()

    bonds_in_ring = [[] for _ in range(len(sssrs))]
    for ringN, bonds in enumerate(bond_rings):
        for bond in bonds:
            bObj = new_rd.GetBondWithIdx(bond)
            bonds_in_ring[ringN].append((bObj.GetBeginAtomIdx(), bObj.GetEndAtomIdx()))
    # print("bonds in ring", bonds_in_ring)
    # print("bond rings", bond_rings, type(bond_rings))

    ring_neighbors_info = {}

    for aID in set([xx for x in atoms_in_ring for xx in x]):
        atom = new_rd.GetAtomWithIdx(aID)
        ringIDs = RingInfo.AtomMembers(aID)
        ring_bonds = [bond for bond in atom.GetBonds() if bond.IsInRing()]
        ring_dict = dict([(i, []) for i in ringIDs])
        for bond in ring_bonds:
            for bRID in RingInfo.BondMembers(bond.GetIdx()):
                ring_dict[bRID].append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        ring_neighbors_info[aID] = ring_dict.values()

    """
    # Example: Benzene
    >>> atoms_in_ring: [[0, 1, 2, 3, 4, 5]]
    >>> bonds_in_ring: [[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]]
    >>> ring_neighbors_info: {0: [[(0, 5), (0, 1)]], 
                              1: [[(1, 2), (0, 1)]], 
                              2: [[(1, 2), (2, 3)]], 
                              3: [[(3, 4), (2, 3)]], 
                              4: [[(3, 4), (4, 5)]], 
                              5: [[(0, 5), (4, 5)]]}
    """

    return atoms_in_ring, bonds_in_ring, ring_neighbors_info


def get_lists(molecule, strict=True):
    # period, group, adj
    period_list, group_list = molecule.get_period_group_list()
    adj_matrix = np.copy(molecule.get_matrix("adj"))
    adj_list = np.sum(adj_matrix, axis=1)

    # neighbor, bond, bond mapping
    neighbor_list = molecule.get_neighbor_list()
    bond_list = molecule.get_bond_list(False)
    bond_mapping = {key: val for val, key in enumerate(bond_list)}

    # valence, atomic number
    ve_list = np.zeros_like(group_list)
    z_list = molecule.get_z_list()
    for i in range(len(group_list)):
        if period_list[i] == 1:
            ve_list[i] = 2
        elif period_list[i] == 2:
            # not considering expanded octet here
            ve_list[i] = 8
        else:
            ve_list[i] = 8 if strict else 2 * group_list[i]
            # ve_list[i] = 18

    # ring membership
    _, _, ring_neighbors_info = get_ring_info(z_list, adj_matrix)

    # electronegativity
    en_list = np.array([EN_TABLE[periodic_table[z - 1]] for z in z_list])

    return (
        period_list,
        group_list,
        z_list,
        ve_list,
        adj_list,
        bond_list,
        bond_mapping,
        neighbor_list,
        ring_neighbors_info,
        en_list,
    )


def get_expanded_ve_list(period_list, group_list, ve_list, chg_list):
    # apply expanded octet rule only if satisfies the following conditions
    # 1. period > 2
    # 2. has non-zero formal charge
    # Ring constraint is not considered here anymore

    # ring_members = np.unique(sum(ring_list, []))
    # in_ring = np.zeros_like(period_list)
    # if len(ring_members) > 0:
    #    in_ring[ring_members] = 1
    # in_ring = in_ring.astype(bool)

    # expanded_idx = (period_list > 2) & ~in_ring
    expanded_idx = period_list > 2
    eve_list = np.copy(ve_list)
    eve_list[expanded_idx] += (
        2 * np.where(chg_list > 0, np.minimum(group_list, chg_list), 0)[expanded_idx]
    )

    return eve_list


def get_modified_list(
    period_list, eve_list, chg_list, bo_dict, ring_list, ring_bond_list
):
    mve_list = np.copy(eve_list)
    ring_members = np.unique(sum(ring_list, []))
    in_ring = np.zeros_like(period_list)
    if len(ring_members) > 0:
        in_ring[ring_members] = 1
    in_ring = in_ring.astype(bool)

    # CleanUp valence expansion
    # subject:
    # period>2, ring member, one or no double/triple bonds with
    # other ring members
    rbtbc = np.zeros_like(period_list)
    for ring_bonds in ring_bond_list:
        for bond in ring_bonds:
            if bo_dict[bond] > 1:
                rbtbc[bond[0]] += 1
                rbtbc[bond[1]] += 1
    cleanUp_idx = (np.array(period_list) > 2) & in_ring & (rbtbc < 2)

    mve_list[cleanUp_idx] += 2 * np.where(chg_list > 0, chg_list, 0)[cleanUp_idx]

    return mve_list


# TODO: Code Acceleration
# Use single Gurobi model for both optimization and resolution


def maximize_bo(
    atom_num,
    bond_num,
    period_list,
    group_list,
    bond_list,
    bond_mapping,
    neighbor_list,
    en_list,
    ring_neighbors_info,
    chg_mol,
    eIsEven,
    **kwargs,
):
    # early stop
    if atom_num == 1:
        return np.array([chg_mol]), {}, (None, None, None)

    ### model construction
    prob = pl.LpProblem("maximize_bo", pl.LpMaximize)

    verbose = kwargs.get("printOptLog", False)
    Xsingle = kwargs.get("HalogenConstraint", False)
    cleanUp = kwargs.get("cleanUp", False) and (len(ring_neighbors_info) > 0)
    M_list  = kwargs.get("MetalCenters", [])

    db_starts = kwargs.get("db_starts", [0] * bond_num)
    tb_starts = kwargs.get("tb_starts", [0] * bond_num)
    t1_starts = kwargs.get("t1_starts", [0] * 2 * atom_num)

    # Bond order
    # db: double bond flag
    # tb: triple bond flag
    # the bond order would be 1, 2, 3
    # from the nature, the constraint db + tb <= 1 should be given
    # and bond order is represented as (1 + db + 2 * tb)

    db = pl.LpVariable.dicts("dbFlag", range(bond_num), 0, 1, pl.LpBinary)
    tb = pl.LpVariable.dicts("tbFlag", range(bond_num), 0, 1, pl.LpBinary)

    prob.extend({f"BondOrderFlag_{i}": db[i] + tb[i] <= 1 for i in range(bond_num)})

    # t1: formal charge
    # t1[2i]: fc+ | t1[2i+1]: fc-
    # t1[2i] - t1[2i+1] : fc of atom i
    # t1[2i] + t1[2i+1] : abs(fc) of atom i
    t1 = pl.LpVariable.dicts("t1", range(2 * atom_num), 0, None, pl.LpInteger)

    # t2: formal charge for weighted objective function
    # weight considering electronegativity

    # o: octet distance
    # the distance between the number of valence elctrons and
    # the octet number(2 for 1st period, 8 for 2nd or higher period)
    # o = 8 - (g - c + b)

    # Set Initial Values
    for i in range(bond_num):
        db[i].setInitialValue(db_starts[i])
        tb[i].setInitialValue(tb_starts[i])
    for i in range(2 * atom_num):
        t1[i].setInitialValue(t1_starts[i])

    # TODO: Revision of Halogen Constraint
    # Halogen atoms, especially Cl and Br, are not allowed for
    # following the extended octet rule.
    # RDKit does not allow Cl and Br to have valence state greater than 1

    # even: dummy variable to force no. of electrons even
    even = pl.LpVariable.dicts("even", range(atom_num), 0, None, pl.LpInteger)

    ### objectives and constraints construction
    # objective functions
    min_od_obj = pl.LpAffineExpression(name="min_od")
    min_fc_obj = pl.LpAffineExpression(name="min_fc")
    max_bo_obj = pl.LpAffineExpression(name="max_bo")
    min_en_obj = pl.LpAffineExpression(name="min_en")

    # constraints
    chg_constr = pl.LpAffineExpression(name="chg_consv")

    for i in range(atom_num):
        lp_constr = pl.LpAffineExpression(name=f"lp_{i}")
        ve_constr = pl.LpAffineExpression(name=f"ve_{i}")

        ve_constr.addInPlace(group_list[i])

        chg_constr.addInPlace(t1[2 * i] - t1[2 * i + 1])
        lp_constr.addInPlace(t1[2 * i] - t1[2 * i + 1])
        ve_constr.addInPlace(-t1[2 * i] + t1[2 * i + 1])
        min_fc_obj.addInPlace(t1[2 * i] + t1[2 * i + 1])
        min_en_obj.addInPlace(en_list[i] * (t1[2 * i] - t1[2 * i + 1]))

        # summation over bond
        for j in neighbor_list[i]:
            a, b = i, j
            if a > b:
                a, b = b, a

            bo = pl.LpAffineExpression(
                1 + db[bond_mapping[(a, b)]] + tb[bond_mapping[(a, b)]] * 2
            )

            lp_constr.addInPlace(bo)
            ve_constr.addInPlace(bo)

            max_bo_obj.addInPlace(bo)

            # halogen atoms have only one single bond
            # halogens might have any bond (halogen anions), and in such case, does not apply the constraint
            if Xsingle and group_list[i] == 7 and period_list[i] <= 4:
                prob += bo == 1, f"XC_{i}"

            # metal constraint
            if i in M_list:
                prob += bo == 1, f"SB_{i}_{j}"

        # the number of lone pair should not be negative
        prob += lp_constr <= group_list[i], f"lp_{i}"

        # octet rule
        # octet distance
        if period_list[i] == 1:
            min_od_obj.addInPlace(2 - ve_constr)
            prob += 2 - ve_constr >= 0, f"od_{i}"
        elif period_list[i] == 2 or len(neighbor_list[i]) <= 4:
            min_od_obj.addInPlace(8 - ve_constr)
            prob += 8 - ve_constr >= 0, f"od_{i}"

        # the number of valence electron is even number (no radical rule!)
        if eIsEven:
            prob += ve_constr == 2 * even[i], f"noRad_{i}"

        # Ring Constraint
        if cleanUp and (i in ring_neighbors_info):
            for n, neighbor_bond_list in enumerate(ring_neighbors_info[i]):
                ring_constr = pl.LpAffineExpression(name=f"ring_{i}_{n}")
                for ring_bond in neighbor_bond_list:
                    ring_constr.addInPlace(
                        db[bond_mapping[ring_bond]] + tb[bond_mapping[ring_bond]]
                    )
                prob += ring_constr <= 1, f"ring_{i}_{n}"

    prob += chg_constr == chg_mol, "chg_consv"

    od_priority = 1  # octet distance priority
    bo_priority = 2  # bond order maximization priority
    chg_priority = 3  # charge separation priority
    en_priority = 4  # electronegativity priority

    if kwargs.get("mode", "") == "fc":
        bo_priority, chg_priority = chg_priority, bo_priority

    objs = [
        (od_priority, min_od_obj, pl.LpMinimize),
        (bo_priority, max_bo_obj, pl.LpMaximize),
        (chg_priority, min_fc_obj, pl.LpMinimize),
        # (en_priority, min_en_obj, pl.LpMinimize),
    ]
    objs = sorted(objs, key=lambda x: x[0])

    # Pulp optimization
    #prob, statuses, objvalues = moSolve(prob, objs, True)
    prob, statuses, objvalues = moSolve(prob, objs, False)

    # error handling
    for i, status in enumerate(statuses):
        if status != pl.LpStatusOptimal:
            print(
                f"maximize_bo: Obj{i} Optimization failed. (status: {pl.LpStatus[status]})",
                file=sys.stderr,
            )
            return None, None, (None, None, None)

    # result record
    if verbose:
        import json

        output = prob.toDict()
        output["status"] = statuses
        output["obj_values"] = objvalues
        json.dump(output, open("output.json", "w"), indent=4, default=str)

    # retrieval
    bo_dict = {}
    chg_list = np.zeros(atom_num, dtype=np.int64)
    for i in range(bond_num):
        bo = 1 + db[i].value() + 2 * tb[i].value()
        bo_dict[bond_list[i]] = int(bo)
    for i in range(atom_num):
        chg_list[i] = int(t1[2 * i].value() - t1[2 * i + 1].value())
    db_values = [int(db[i].value()) for i in range(bond_num)]
    tb_values = [int(tb[i].value()) for i in range(bond_num)]
    t1_values = [int(t1[i].value()) for i in range(2 * atom_num)]

    return chg_list, bo_dict, (db_values, tb_values, t1_values)


def resolve_chg(
    atom_num,
    bond_num,
    period_list,
    group_list,
    bond_list,
    bond_mapping,
    neighbor_list,
    en_list,
    ring_neighbors_info,
    chg_mol,
    eIsEven,
    overcharged,
    db_starts,
    tb_starts,
    t1_starts,
    stepIdx=0,
    **kwargs,
):

    if atom_num == 1:
        return np.array([chg_mol]), {}

    ### model construction
    prob = pl.LpProblem(f"resolve_chg{stepIdx}", pl.LpMaximize)

    verbose = kwargs.get("printOptLog", False)
    Xsingle = kwargs.get("HalogenConstraint", False)
    cleanUp = kwargs.get("cleanUp", False) and (len(ring_neighbors_info) > 0)
    M_list = kwargs.get("MetalCenters", [])

    # bo: bond order
    db = pl.LpVariable.dicts("dbFlag", range(bond_num), 0, 1, pl.LpBinary)
    tb = pl.LpVariable.dicts("tbFlag", range(bond_num), 0, 1, pl.LpBinary)
    prob.extend({f"BondOrderFlag_{i}": db[i] + tb[i] <= 1 for i in range(bond_num)})

    # t1: formal charge
    t1 = pl.LpVariable.dicts("t1", range(2 * atom_num), 0, None, pl.LpInteger)

    # t2: formal charge for weighted objective function
    # weight considering electronegativity
    # t2 = model.addVars(2 * atom_num, name="t2", vtype=GRB.CONTINUOUS)

    # Set Initial Values
    for i in range(bond_num):
        db[i].setInitialValue(db_starts[i])
        tb[i].setInitialValue(tb_starts[i])
    for i in range(2 * atom_num):
        t1[i].setInitialValue(t1_starts[i])

    # even: dummy variable to force no. of electrons even
    even = pl.LpVariable.dicts("even", range(atom_num), 0, None, pl.LpInteger)

    ### objectives and constraints construction
    # objective functions
    min_fc_obj = pl.LpAffineExpression(name="min_fc")
    max_bo_obj = pl.LpAffineExpression(name="max_bo")
    min_en_obj = pl.LpAffineExpression(name="min_en")
    # constraints
    chg_constr = pl.LpAffineExpression(name="chg_consv")

    for i in range(atom_num):
        lp_constr = pl.LpAffineExpression(name=f"lp_{i}")
        ve_constr = pl.LpAffineExpression(name=f"ve_{i}")
        X_flag = Xsingle and group_list[i] == 7 and period_list[i] <= 4

        ve_constr.addInPlace(group_list[i])
        prev_ve = group_list[i]  # previous valence electron

        chg_constr.addInPlace(t1[2 * i] - t1[2 * i + 1])
        lp_constr.addInPlace(t1[2 * i] - t1[2 * i + 1])
        ve_constr.addInPlace(-t1[2 * i] + t1[2 * i + 1])
        min_fc_obj.addInPlace(t1[2 * i] + t1[2 * i + 1])
        min_en_obj.addInPlace(en_list[i] * (t1[2 * i] - t1[2 * i + 1]))
        prev_ve += -t1_starts[2 * i] + t1_starts[2 * i + 1]  # previous valence electron

        # summation over bond
        for j in neighbor_list[i]:
            a, b = i, j
            if a > b:
                a, b = b, a

            bo = pl.LpAffineExpression(
                1 + db[bond_mapping[(a, b)]] + tb[bond_mapping[(a, b)]] * 2
            )

            lp_constr.addInPlace(bo)
            ve_constr.addInPlace(bo)

            max_bo_obj.addInPlace(bo)

            prev_ve += (
                1
                + db_starts[bond_mapping[(a, b)]]
                + 2 * tb_starts[bond_mapping[(a, b)]]
            )  # previous valence electron
            # Halogen Constraint
            # halogen atoms should obey the octet rule
            # (no extended octet rule for halogens)
            # TODO: Revision of Halogen Constraint
            # Halogen atoms, especially Cl and Br, are not allowed for
            # following the extended octet rule.
            # RDKit does not allow Cl and Br to have valence state greater than 1

            # halogen atoms have only one single bond
            # halogens might not have any bond (halogen anions), and in such case, does not apply the constraint
            if Xsingle and group_list[i] == 7 and period_list[i] <= 4:
                prob += bo == 1, f"XC_{i}"

            # metal constraint
            if i in M_list:
                prob += bo == 1, f"SB_{i}_{j}"

        # the number of lone pair should not be negative
        prob += lp_constr <= group_list[i], f"lp_{i}"

        # octet rule
        # if charged and period > 2, apply expanded octet rule
        # else, freeze the valence (octet rule)
        if not bool(overcharged[i]):
            prob += (
                ve_constr == prev_ve,
                f"ve_freeze_{i}",
            )  # don't know why this is not working
            # prob += ve_constr <= prev_ve, f"ve_freeze_{i}"  # this constraint goes wrong with azide moiety
        else:
            prob += ve_constr >= prev_ve, f"ve_expanded_{i}"

        # the number of valence electron is even number (no radical rule!)
        if eIsEven:
            prob += ve_constr == 2 * even[i], f"noRad_{i}"

        # Ring Constraint
        if cleanUp and (i in ring_neighbors_info):
            for n, neighbor_bond_list in enumerate(ring_neighbors_info[i]):
                ring_constr = pl.LpAffineExpression(name=f"ring_{i}_{n}")
                for ring_bond in neighbor_bond_list:
                    ring_constr.addInPlace(
                        db[bond_mapping[ring_bond]] + tb[bond_mapping[ring_bond]]
                    )
                prob += ring_constr <= 1, f"ring_{i}_{n}"

    prob += chg_constr == chg_mol, "chg_consv"

    ### optimization
    bo_priority = 2  # bond order maximization priority
    chg_priority = 1  # charge separation priority
    en_priority = 3  # electronegativity priority

    objs = [
        # (bo_priority, max_bo_obj, pl.LpMaximize),
        (chg_priority, min_fc_obj, pl.LpMinimize),
        (en_priority, min_en_obj, pl.LpMinimize),
    ]
    objs = sorted(objs, key=lambda x: x[0])

    # Pulp optimization
    #prob, statuses, objvalues = moSolve(prob, objs, True)
    prob, statuses, objvalues = moSolve(prob, objs, False)

    # error handling
    for i, status in enumerate(statuses):
        if status != pl.LpStatusOptimal:
            print(
                f"resolve_chg: Obj{i} Optimization failed. (status: {pl.LpStatus[status]})",
                file=sys.stderr,
            )
            return None, None, (None, None, None)

    # result record
    if verbose:
        import json

        output = prob.toDict()
        output["status"] = statuses
        output["obj_values"] = objvalues
        json.dump(
            output, open(f"output_resolve{stepIdx}.json", "w"), indent=4, default=str
        )

    # retrieval
    bo_dict = {}
    chg_list = np.zeros(atom_num, dtype=np.int64)
    for i in range(bond_num):
        bo = 1 + db[i].value() + 2 * tb[i].value()
        bo_dict[bond_list[i]] = int(bo)
    for i in range(atom_num):
        chg_list[i] = int(t1[2 * i].value() - t1[2 * i + 1].value())
    db_values = [int(db[i].value()) for i in range(bond_num)]
    tb_values = [int(tb[i].value()) for i in range(bond_num)]
    t1_values = [int(t1[i].value()) for i in range(2 * atom_num)]

    return chg_list, bo_dict, (db_values, tb_values, t1_values)


def compute_chg_and_bo_debug(molecule, chg_mol, resolve=True, cleanUp=True, **kwargs):
    (
        period_list,
        group_list,
        z_list,
        ve_list,
        adj_list,
        bond_list,
        bond_mapping,
        neighbor_list,
        ring_neighbors_info,
        en_list,
    ) = get_lists(molecule)

    atom_num, bond_num = len(z_list), len(bond_list)
    eIsEven = int(np.sum(z_list) - chg_mol) % 2 == 0
    resolve_step = 0
    kwargs["cleanUp"] = cleanUp

    chg_list0, bo_dict0, raw_outputs = maximize_bo(
        atom_num,
        bond_num,
        period_list,
        group_list,
        bond_list,
        bond_mapping,
        neighbor_list,
        en_list,
        ring_neighbors_info,
        chg_mol,
        eIsEven,
        **kwargs,
    )

    # early stop
    if chg_list0 is None and bo_dict0 is None:
        chg_list0, bo_matrix0 = None, None
    else:
        bo_matrix0 = np.zeros((atom_num, atom_num))
        for p, q in bo_dict0.keys():
            bo_matrix0[p][q] = bo_dict0[(p, q)]
            bo_matrix0[q][p] = bo_dict0[(p, q)]

    # check charge separation
    chg_sep = np.any(chg_list0 > 0) and np.any(chg_list0 < 0)

    bo_matrix1, chg_list1 = np.copy(bo_matrix0), np.copy(chg_list0)  # place holder

    # charge resolution
    if resolve and chg_sep:
        print("Debug: resolution")
        bo_sum = np.zeros(atom_num)
        for p, q in bo_dict0.keys():
            bo_sum[p] += bo_dict0[(p, q)]
            bo_sum[q] += bo_dict0[(p, q)]

        # TODO: Check the condition for overcharged atoms
        # 1. period > 2
        # 2. non-zero charge on itself
        overcharged = (period_list > 2) & (np.abs(chg_list0) != 0)
        print("Debug: overcharged", np.nonzero(overcharged))

        chg_list1, bo_dict1, raw_outputs1 = resolve_chg(
            atom_num,
            bond_num,
            period_list,
            group_list,
            bond_list,
            bond_mapping,
            neighbor_list,
            en_list,
            ring_neighbors_info,
            chg_mol,
            eIsEven,
            overcharged,
            raw_outputs[0],
            raw_outputs[1],
            raw_outputs[2],
            **kwargs,
        )

        resolve_step += 1

        # error handling
        if bo_dict1 is None and chg_list1 is None:
            chg_list1, bo_matrix1 = None, None
        else:
            bo_matrix1 = np.zeros((atom_num, atom_num))
            for p, q in bo_dict1.keys():
                bo_matrix1[p][q] = bo_dict1[(p, q)]
                bo_matrix1[q][p] = bo_dict1[(p, q)]

    return chg_list0, bo_matrix0, chg_list1, bo_matrix1


def compute_chg_and_bo(
    molecule, chg_mol, resolve=True, cleanUp=True, **kwargs
):
    """
    Compute the charge and bond order for a given molecule.

    Args:
        molecule (Molecule): The molecule object containing atomic and bonding information.
        chg_mol (int): The total charge of the molecule.
        resolve (bool, optional): Whether to go through charge resolution step if needed. Defaults to True.
        cleanUp (bool, optional): Whether to apply heuristics that cleans up the resulting molecular graph. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to the maximize_bo and resolve_chg functions.

    Returns:
        tuple: A tuple containing the list of charges for each atom and the bond order matrix.::

    """

    (
        period_list,
        group_list,
        z_list,
        ve_list,
        adj_list,
        bond_list,
        bond_mapping,
        neighbor_list,
        ring_neighbors_info,
        en_list,
    ) = get_lists(molecule)

    atom_num, bond_num = len(z_list), len(bond_list)
    eIsEven = int(np.sum(z_list) - chg_mol) % 2 == 0
    resolve_step = 0
    kwargs["cleanUp"] = cleanUp

    chg_list, bo_dict, raw_outputs = maximize_bo(
        atom_num,
        bond_num,
        period_list,
        group_list,
        bond_list,
        bond_mapping,
        neighbor_list,
        en_list,
        ring_neighbors_info,
        chg_mol,
        eIsEven,
        **kwargs,
    )

    # early stop
    if bo_dict is None and chg_list is None:
        return None, None

    # check charge separation
    chg_sep = np.any(chg_list > 0) and np.any(chg_list < 0)

    # charge resolution
    if resolve and chg_sep:
        bo_sum = np.zeros(atom_num)
        for p, q in bo_dict.keys():
            bo_sum[p] += bo_dict[(p, q)]
            bo_sum[q] += bo_dict[(p, q)]

        # TODO: Check the condition for overcharged atoms
        # 1. period > 2
        # 2. non-zero charge on itself
        overcharged = (period_list > 2) & (np.abs(chg_list) != 0)

        chg_list, bo_dict, raw_outputs2 = resolve_chg(
            atom_num,
            bond_num,
            period_list,
            group_list,
            bond_list,
            bond_mapping,
            neighbor_list,
            en_list,
            ring_neighbors_info,
            chg_mol,
            eIsEven,
            overcharged,
            raw_outputs[0],
            raw_outputs[1],
            raw_outputs[2],
            stepIdx=resolve_step,
            **kwargs,
        )

        resolve_step += 1

        # error handling
        if bo_dict is None and chg_list is None:
            return None, None

    bo_matrix = np.zeros((atom_num, atom_num))
    for p, q in bo_dict.keys():
        bo_matrix[p][q] = bo_dict[(p, q)]
        bo_matrix[q][p] = bo_dict[(p, q)]

    return chg_list, bo_matrix


if __name__ == "__main__":
    import sys

    smi = sys.argv[1]
