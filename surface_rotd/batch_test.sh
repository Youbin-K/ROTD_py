#!/usr/bin/bash

#SBATCH --job-name=ROTD_distance_check
#SBATCH --time=100:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=batch
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=cfgoldsm-condo
#SBATCH --mail-type=END
#SBATCH --mail-user=youbin_kim@brown.edu

source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate py3.8
#conda activate python3.7_test

module load mpi/openmpi_3.1.6_gcc
module load ase/3.13.0
#module load ase/3.19.1

which python3
#mpiexec -n ${SLURM_NTASKS} python3 get_attribute.py
#mpiexec -n ${SLURM_NTASKS} python3 ch3ch3_close.py
mpiexec -n ${SLURM_NTASKS} python3 copt.py


