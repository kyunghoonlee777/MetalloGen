module load library_module/intel_mkl_amd0
module load compiler_module/intel_amd0
module load software_module/g09.shchoi
module load mpi_module/openmpi-4.1.2-intel

export PATH=/home/rxn_grp/programs/:/appl/orca_6_0_1_linux_x86-64_shared_openmpi416/:$PATH
export ORCA=/appl/orca_6_0_1_linux_x86-64_shared_openmpi416/orca
export xtbbin=/home/rxn_grp/programs/xtb-gaussian

source activate metallogen