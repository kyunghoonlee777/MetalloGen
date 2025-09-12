# MetalloGen

MetalloGen is a Python package for generating 3D structures of organometallic complexes.

---

## Installation

We recommend using **conda** for RDKit installation.

```bash
# Clone the repository
git clone https://github.com/<USERNAME>/MetalloGen.git
cd MetalloGen

# Create environment (example: Python 3.10)
conda create -n metallogen python=3.10 -y
conda activate metallogen

# Install RDKit (recommended fixed version for reproducibility)
conda install -c conda-forge rdkit=2023.03.2 -y

# Install MetalloGen
pip install -e .
```

---

## Usage

You can run MetalloGen either via the installed console script:

```bash
metallogen -s "[Ir+]|CP:1C|CP:2C|[Cl-:3]|[C-:4]#[O+]|4_square_planar" \
           -wd "working_directory" \
           -sd "save_directory" \
           -r 1
```

or via the Python module:

```bash
python -m MetalloGen -s "[Ir+]|CP:1C|CP:2C|[Cl-:3]|[C-:4]#[O+]|4_square_planar" \
                     -wd "working_directory" \
                     -sd "save_directory" \
                     -r 1
```

---

## Command-line Arguments

The following options are available:

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--smiles` | `-s` | `str` | *required* | Input MSMILES string |
| `--working_directory` | `-wd` | `str` | `None` | Scratch directory for running quantum chemical calculation |
| `--save_directory` | `-sd` | `str` | `None` | Directory to save the results |
| `--final_relax` | `-r` | `int` | `1` | Whether to perform final relaxation after generation (`0` = no, `1` = yes) |

---

## Example

```bash
metallogen -s "[Ir+]|CP:1C|CP:2C|[Cl-:3]|[C-:4]#[O+]|4_square_planar" \
           -wd "./scratch" \
           -sd "./results" \
           -r 1
```

---

## Output

Results will be saved in the directory specified by `--save_directory`.  
Typical outputs include:

- Optimized 3D coordinates (`.xyz`, `.mol`, or `.sdf`)
- Logs from quantum chemical calculations
- Final relaxed structure (if `-r 1` is set)

---

## Figures

You can include sample output figures here (for example, molecular structures or energy profiles):


---

## License

This project is licensed under the BSD 3-Clause License.
