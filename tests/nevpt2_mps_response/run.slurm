#!/bin/bash
#SBATCH --job-name=F_no4rdm_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --mem=200G
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --output=tests/nevpt2_mps_response/slurm-%j.out

set -euo pipefail
cd /home/Yxwxwx/code/socutils
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
exec bash tests/nevpt2_mps_response/run.sh
