#!/bin/bash -l
#SBATCH --job-name=velocity-h_0.67556
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --account=go25
#SBATCH --output=slurm-h_0.67556.out
#SBATCH --error=slurm-h_0.67556.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash -lc '
module load cray-mpich
module load cuda
module load fftw
module load gcc
module load gsl
module load hdf5

export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDZV_THRESHOLD=2097152
export FI_CXI_REQ_BUF_SIZE=4194304
export FI_CXI_REQ_BUF_MIN_POSTED=64
export FI_CXI_REQ_BUF_MAX_CACHED=256
export OMP_NUM_THREADS=72
export OMP_PLACES=cores

cd /capstor/scratch/cscs/leporif/VelocityField/gevolution-2.0/
srun -N 4 -n 16 --cpus-per-task=72 -C gpu -A go25 --time=00:30:00 --partition=normal --hint=exclusive --cpu-bind=socket ./gpu-bind.sh ./gevolution -n 4 -m 4 -s /capstor/scratch/cscs/leporif/VelocityField/gevolution-2.0/output/grad-1/h-deriv/Ngrid512_Lbox128//h_0.67556_As_2.215e-9/settings.ini
'
