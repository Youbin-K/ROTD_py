#!/usr/bin/bash

#SBATCH --job-name=ROTD_19_long_full
#SBATCH --time=100:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=batch
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=cfgoldsm-condo
#SBATCH --mail-type=END,fail
#SBATCH --mail-user=youbin_kim@brown.edu

# This is REDHAT 9 ANACONDA
#source /oscar/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/anaconda-2023.09-0-7nso27ys7navjquejqdxqylhg7kuyvxo/etc/profile.d/conda.sh
#conda activate rotd_env_ase3.19


# This is REDHAT 9 MINICONDA
#export OMPI_MCA_btl=^openib
#source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
#module load miniconda3/23.11.0s-odstpk5
#conda activate test9_yml
#module load hpcx-mpi/4.1.5rc2s-yflad4v # Fails from the MPI init process
#echo $CONDA_DEFAULT_ENV$


#module load openmpi/4.1.2-s5wtoqb # At least loads AMP. But does not run
#PATH=$PATH:/oscar/runtime/software/external/openmpi/4.0.7/src/openmpi-4.0.7/bin
#conda activate new_env_from_RH7 #This has numpy version of 1.24.4 #Does not work with np.int problem

# This is REDHAT 9 PYTHON env
export OMPI_MCA_btl=^openib
source /users/ykim219/virtual_env/test_env/bin/activate
module load hpcx-mpi/4.1.5rc2s-yflad4v # Fails from the MPI init process
#module load openmpi/4.1.2-s5wtoqb # At least loads AMP. But does not run
#PATH=$PATH:/oscar/runtime/software/external/openmpi/4.0.7/src/openmpi-4.0.7/bin


# This is for REDHAT 7. 
#source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh # RH 9 conda does not work in RH7
#source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
#module load mpi/openmpi_4.0.7_gcc_10.2_slurm22
#conda activate test2
#conda activate test9_yml

which python3
which ase

#mpiexec -n ${SLURM_NTASKS} python3 ch3ch3_long.py
mpiexec -n ${SLURM_NTASKS} --bind-to core:overload-allowed python3 ch3ch3_long.py
#srun --mpi=pmix python3 ch3ch3_long.py


