#!/usr/bin/bash

#SBATCH --job-name=ROTD_surface_test
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=batch
##SBATCH --partition=debug
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=cfgoldsm-condo
#SBATCH --mail-type=END,fail
#SBATCH --mail-user=youbin_kim@brown.edu

# This is REDHAT 9 MINICONDA
#export OMPI_MCA_btl=^openib
#module load miniconda3/23.11.0s-odstpk5
#source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
#module load openmpi/4.1.5-hkgv3gi
#conda activate rotd_rhel9_conda


# This is REDHAT 9 PYTHON env
export OMPI_MCA_btl=^openib
source /users/ykim219/virtual_env/test_env/bin/activate # ASE 3.13.0 Currently using, since I cannot see the labframe with 3.19.1
#source /users/ykim219/virtual_env/working_rotd_rhel9/bin/activate # ASE 3.19.1
module load hpcx-mpi/4.1.5rc2s-yflad4v # Fails from the MPI init process


#which python3
#which ase

# RHEL7 run Command
#mpiexec -n ${SLURM_NTASKS} python copt.py

# RHEL 9 RUN COMMAND
#mpiexec -n ${SLURM_NTASKS} --bind-to core:overload-allowed python3 ch3ch3_long.py
mpiexec -n ${SLURM_NTASKS} --bind-to core:overload-allowed python3 copt.py

#srun --mpi=pmix python3 ch3ch3_long.py


