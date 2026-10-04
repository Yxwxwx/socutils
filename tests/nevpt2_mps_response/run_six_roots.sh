#!/bin/bash
set -euo pipefail
ulimit -s unlimited
export OMP_STACKSIZE=256M
module purge
module load anaconda3-2024.10-1 openmpi-5.0.6/gcc-13.3.0
cd /home/Yxwxwx/code/socutils
mkdir -p /nvme/Yxwxwx
task_scratch=$(mktemp -d /nvme/Yxwxwx/F_sa6_all_roots.XXXXXX)
trap 'rm -rf -- "$task_scratch"' EXIT
export TMPDIR="$task_scratch" PYSCF_TMPDIR="$task_scratch"
# Two separate Block2 processes: 16 threads each, 32 CPUs total.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
export PYSCF_MAX_MEMORY=200000 PYTHONFAULTHANDLER=1
unset PYTHONPATH
printf 'Host=%s PID=%s workers=2 threads_per_worker=16 scratch=%s\n' "$(hostname)" "$$" "$task_scratch"
exec_python=/home/Yxwxwx/code/socutils/.venv/bin/python
"$exec_python" -u -m tests.nevpt2_mps_response.six_roots "$@"
"$exec_python" -u -m tests.nevpt2_mps_response.six_roots --summary-only "$@"
