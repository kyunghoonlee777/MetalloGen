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

You can run MetalloGen either via the installed console script:

```jsx
metallogen -s "[Ir+]|CP:1C|CP:2C|[Cl-:3]|[C-:4]#[O+]|4_square_planar" \
           -wd "working_directory" \
           -sd "save_directory" \
           -r 1
```

or via the Python module:

```jsx
python -m MetalloGen -s "[Ir+]|CP:1C|CP:2C|[Cl-:3]|[C-:4]#[O+]|4_square_planar" \
                     -wd "working_directory" \
                     -sd "save_directory" \
                     -r 1
```

---

# Command-line Arguments

The following options are available:

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--smiles` | `-s` | `str` | *required* | Input MSMILES string |
| `--working_directory` | `-wd` | `str` | `None` | Scratch directory for running quantum chemical calculation |
| `--save_directory` | `-sd` | `str` | `None` | Directory to save the results |
| `--final_relax` | `-r` | `int` | `1` | Whether to perform final relaxation after generation (`0` = no, `1` = yes) |

---

# Output

Results will be saved in the directory specified by `--save_directory`.  
Typical outputs include:

- Optimized 3D coordinates (`.xyz`, `.mol`, or `.sdf`)
- Logs from quantum chemical calculations
- Final relaxed structure (if `-r 1` is set)

---

# License

This project is licensed under the BSD 3-Clause License.

---

# Contact Information

Please e-mail me to here: [kyunghoonlee@kaist.ac.kr](mailto:kyunghoonlee@kaist.ac.kr) for more detail discussion
