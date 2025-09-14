# MetalloGen

A Python package for generating 3D structures of organometallic complexes.

---

# Requirements

- Python ≥ 3.9  
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [cclib](https://cclib.github.io/)
- [RDKit](https://www.rdkit.org/)
- [PuLP](https://coin-or.github.io/pulp/)

---

# Settings

Before using MetalloGen, following things should be prepared:

1. Quantum chemistry (QC) package should be configured. For example, if users want to use Guassian, either 'which g09', or 'which g16' should be correctly identified, as following:

    ```jsx
    >> which g09
    >> /appl/g09.shchoi/G09Files/g09/g09

    >> which xtb
    >> /home/rxn_grp/programs/xtb
    ```

2. If you want to use our **default method ('xtb-gaussian')**, you must set an environment variable **'xtbbin'** should be specified as following:

    - You can obtain `xtb-gaussian` from the [Aspuru-Guzik Group GitHub repository](https://github.com/aspuru-guzik-group/xtb-gaussian).  
    - After installation, set the environment variable:
    ```jsx
    >> export xtbbin="/home/rxn_grp/programs/xtb-gaussian"
    ```
    - Verify:
    ```jsx
    >> echo $xtbbin
    >> /home/rxn_grp/programs/xtb-gaussian
    ```

# Installation

```jsx
# Clone the repository
>> git clone https://github.com/kyunghoonlee777/MetalloGen.git
>> cd MetalloGen

# Create environment
>> conda create -n metallogen python=3.9 -y
>> conda activate metallogen

# Install MetalloGen
>> pip install -e .
```

---

## Executing MetalloGen

MetalloGen uses a modified SMILES representation for **mononuclear coordination complexes**, called **m-SMILES**, as input. From an m-SMILES string, MetalloGen generates the corresponding 3D conformers.

The m-SMILES representation encodes:
- the **metal center** (e.g., `[Zr+4]`),
- the **ligands** as SMILES strings separated by vertical bars (`|`),
- the **coordination geometry** (e.g., `5_trigonal_bipyramidal`).

Donor atoms directly coordinated to the metal are specified with square brackets, and coordination sites are indicated with atom mapping numbers (e.g., `[Cl-:2]` means a chloro ligand bound at coordination site 2).  
This representation makes it straightforward to encode complex polydentate and polyhapto ligands while preserving coordination geometry and stereochemistry.

You can run MetalloGen either via the installed console script:

```jsx
metallogen -s "[Zr+4]|[Cl-:2]|[Cl-:3]|[N:1]1=C(C[C-:4]2[CH:4]=[CH:4][CH:4]=[CH:4]2)C=CC=C1(C[C-:5]3[CH:5]=[CH:5][CH:5]=[CH:5]3)|5_trigonal_bipyramidal" -wd <WORKING DiRECTORY> -sd <SAVE DIRECTORY> -r 1
```

---

# Output

Results will be saved in the directory specified by `--save_directory`.  
Typical outputs include:

- Optimized 3D coordinates (`.xyz`, `.mol`, or `.sdf`)
- Logs from quantum chemical calculations
- Final relaxed structure (if `-r 1` is set)

---

# Command-line Arguments

The following options are available:

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--smiles` | `-s` | `str` | *required* | Input MSMILES string |
| `--working_directory` &nbsp;&nbsp;&nbsp;&nbsp;| `-wd` | `str` | `None` | Scratch directory for running quantum chemical calculation |
| `--save_directory` | `-sd` | `str` | `None` | Directory to save the results |
| `--final_relax` | `-r` | `int` | `1` | Whether to perform final relaxation after generation (`0` = no, `1` = yes) |

---

# License

This project is licensed under the BSD 3-Clause License.

---

# Contact Information

Please e-mail me to here: [kyunghoonlee@kaist.ac.kr](mailto:kyunghoonlee@kaist.ac.kr) for more detail discussion
