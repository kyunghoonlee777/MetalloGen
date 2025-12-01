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

Before using MetalloGen, following must be prepared.

## 1. Quantum chemistry backends
MetalloGen delegates energy and geometry calculations to an external quantum chemistry (QC) backend.  
Currently supported binaries are:

- **Gaussian** (`g09` or `g16`)
- **ORCA** (`orca`)
- **xTB** (`xtb`, for the default `xtb_gaussian` workflow)

Make sure these executables are on your `PATH`. For example:

```bash
>> which g09
/appl/g09.shchoi/G09Files/g09/g09

>> which g16
/appl/g16.shchoi/G16Files/g16/g16

>> which orca
/appl/orca_6_0_1_linux_x86-64_shared_openmpi416/orca

>> which xtb
/home/rxn_grp/programs/xtb
```

## 2. Default `xtb_guassian` calculator

By default, MetalloGen uses the **`xtb_gaussian`** method (`--calculator xtb_gaussian`), which couples xTB with Gaussian via the `xtb-gaussian` wrapper.

- You can obtain `xtb-gaussian` from the  
  [Aspuru-Guzik Group GitHub repository](https://github.com/aspuru-guzik-group/xtb-gaussian).  
- After installation, set the environment variable **`xtbbin`** to the `xtb-gaussian` executable:

```bash
>> export xtbbin="/home/rxn_grp/programs/xtb-gaussian"
```

- Verify:

```bash
>> echo $xtbbin
/home/rxn_grp/programs/xtb-gaussian
```

> **Note:** When `--calculator xtb_gaussian` (the default) is used, both `xtb` and a Gaussian binary (`g09` or `g16`) **and** the `xtbbin` environment variable must be correctly configured.

## 3. Optional ORCA-only calculator

MetalloGen can also use **ORCA directly** as the QC backend:

- Set `--calculator orca` (or `-c orca`).
- Only `orca` is required on your `PATH`; `xtbbin` is **not** needed for this mode.

Example:

```bash
>> which orca
/home/rxn_grp/programs/orca/orca
```

# Installation

```bash
# Clone the repository
>> git clone https://github.com/kyunghoonlee777/MetalloGen.git
>> cd MetalloGen

# Create environment
>> conda create -n metallogen python=3.9 -y
>> conda activate metallogen

# Install MetalloGen (editable mode)
>> pip install -e .
```

---

# Executing MetalloGen

MetalloGen can be executed with two types of inputs:

1. **m-SMILES representation** (modified SMILES for mononuclear coordination complexes)  
2. **MOL/SDF files** containing predefined molecular structures

Internally, MetalloGen uses a **calculator backend** selected via `--calculator` / `-c`:

- `xtb_gaussian` (default): wrapper using the `xtb-gaussian` script (xTB + Gaussian).
- `orca`: direct ORCA calculations.

---

## 1. Using m-SMILES

MetalloGen uses a modified SMILES representation called **m-SMILES** as input. From an m-SMILES string, MetalloGen generates the corresponding 3D conformers.

<p align="center">
  <img src="figures/msmiles.png" alt="m-SMILES encoding" width="600">
</p>

The m-SMILES representation encodes:

- the **metal center** (e.g., `[Zr+4]`)
- the **ligands** as SMILES strings separated by vertical bars (`|`)
- the **coordination geometry** (e.g., `5_trigonal_bipyramidal`)

Donor atoms directly coordinated to the metal are specified with square brackets, and coordination sites are indicated with atom mapping numbers (for example, `[Cl-:2]` means a chloro ligand bound at coordination site 2).  
This makes it straightforward to encode polydentate and polyhapto ligands while preserving coordination geometry and stereochemistry.

**Example (m-SMILES input with default calculator):**

```bash
metallogen \
  -s "[Zr+4]|[Cl-:2]|[Cl-:3]|[N:1]1=C(C[C-:4]2[CH:4]=[CH:4][CH:4]=[CH:4]2)C=CC=C1(C[C-:5]3[CH:5]=[CH:5][CH:5]=[CH:5]3)|5_trigonal_bipyramidal" \
  -wd <WORKING DIRECTORY> \
  -sd <SAVE DIRECTORY> \
  -r 1 \
  -nc 1 \
  -c xtb_gaussian
```

The generated 3D conformer corresponding to the m-SMILES input is shown below:

<p align="center">
  <img src="figures/msmiles_output.png" alt="MetalloGen output conformer" width="400">
</p>

---

## 2. Using MOL/SDF files

In some cases—such as **benchmarking with CSD (Cambridge Structural Database)**—obtaining an m-SMILES representation can be challenging or impractical.  
For these situations, MetalloGen can directly take **MOL** or **SDF** files as input via the `-id` flag. This allows seamless use of existing 3D structures extracted from databases.

As an example, consider a complex extracted from the CSD with refcode **`CIXDAS`**.  
The corresponding 3D structure (in SDF format) can be provided directly to MetalloGen:

<p align="center"> 
  <img src="figures/sdf.png" alt="CSD-extracted SDF structure" width="400"> 
</p>

**Example (SDF input using ORCA):**

```bash
metallogen \
  -id <INPUT DIRECTORY> \
  -wd <WORKING DIRECTORY> \
  -sd <SAVE DIRECTORY> \
  -r 1 \
  -nc 1 \
  -c orca
```

MetalloGen successfully generates well-formed conformers from such SDF inputs as well:

<p align="center"> 
  <img src="figures/sdf_output.png" alt="MetalloGen output from SDF" width="600"> 
</p>

---

# Output

When running MetalloGen, two types of output are generated:

## 1. Conformers

- For each input structure, the number of conformers specified by `--num_conformer` / `-nc` are generated.
- These conformers differ by the initial embedding conditions used in the generation procedure.
- **Each conformer is saved in the directory specified by `--save_directory` / `-sd` as:**

  - `result_{i}.xyz` &nbsp;&nbsp; (where `i = 0, 1, 2, ...`)

- The XYZ files contain full 3D coordinates of the metal complex and can be directly opened in standard molecular viewers.

## 2. Quantum chemical calculation files

MetalloGen calls the selected QC backend (`xtb_gaussian` or `orca`) during generation and relaxation:

- **Working directory (`--working_directory` / `-wd`):**

  - Intermediate **input and output** files for the calculator are written here as scratch.
  - This typically includes calculator-specific files such as Gaussian or ORCA input/output and any temporary files created during optimization steps.
  - You can inspect these files for debugging or detailed analysis; they can also be cleaned up after the run.

- **Save directory (`--save_directory` / `-sd`):**

  - If `--final_relax` / `-r` is set to `1` (default), both the **final relaxation input files** and **corresponding log/output files** are saved alongside the final conformer XYZ files.
  - This means that, for each conformer, the **final 3D structure (`result_{i}.xyz`) and its QC calculation logs** live together in one place, making it easy to track which calculation produced which structure.

> In summary:
>
> - **Geometry & coordinates:** `result_{i}.xyz` in `--save_directory`  
> - **QC scratch/intermediate files:** in `--working_directory`  
> - **Final QC input/output for relaxed structures:** in `--save_directory` (when `-r 1`)

---

# Command-line Arguments

The following options are available:

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--smiles` | `-s` | `str` | `None` | Input m-SMILES string |
| `--input_directory` | `-id` | `str` | `None` | Input SDF/MOL file directory (all files in the directory are processed) |
| `--working_directory` | `-wd` | `str` | `None` | Scratch directory for running quantum chemical calculations |
| `--save_directory` | `-sd` | `str` | `None` | Directory to save final conformers and, optionally, final QC inputs/logs |
| `--final_relax` | `-r` | `int` | `1` | Whether to perform final relaxation after generation (`0` = no, `1` = yes) |
| `--num_conformer` | `-nc` | `int` | `1` | Number of conformers to generate for each input |
| `--calculator` | `-c` | `str` | `xtb_gaussian` | Calculator backend to use: `xtb_gaussian` (default, uses `xtb-gaussian` + Gaussian) or `orca` (direct ORCA) |

> If `--calculator` is omitted, MetalloGen uses `xtb_gaussian` by default.  
> To use ORCA, specify `-c orca` and ensure `orca` is available on `PATH`.

---
# Citation
Please cite as below Kyunghoon Lee, Shinyoung Park, Minseong Park, and Woo Youn Kim. "MetalloGen: Automated 3D Conformer Generation for Diverse Coordination Complexes" Journal of Chemical Information and Modeling 65 (2025): 11878–11891.

---

# License

This project is licensed under the BSD 3-Clause License.

---

# Contact Information

For questions, issues, or collaboration, please contact:

- Kyunghoon Lee - [kyunghoonlee@kaist.ac.kr](mailto:kyunghoonlee@kaist.ac.kr)
- Minseong Park - [pms131131@kaist.ac.kr](mailto:pms131131@kaist.ac.kr)
