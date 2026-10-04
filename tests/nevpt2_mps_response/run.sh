#!/bin/bash
# Direct CPU execution (also reusable inside a Slurm allocation).
set -euo pipefail
# X2CAMF's radial integral work arrays exceed the default 8 MiB stack.
ulimit -s unlimited
export OMP_STACKSIZE=256M
module purge
module load anaconda3-2024.10-1 openmpi-5.0.6/gcc-13.3.0
cd /home/Yxwxwx/code/socutils
mkdir -p /nvme/Yxwxwx
task_scratch=$(mktemp -d /nvme/Yxwxwx/F_sa6_no4rdm_${SLURM_JOB_ID:-cpu}.XXXXXX)
trap 'rm -rf -- "$task_scratch"' EXIT
export TMPDIR="$task_scratch" PYSCF_TMPDIR="$task_scratch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYSCF_MAX_MEMORY=200000
export PYTHONFAULTHANDLER=1
# The venv's editable socutils install already resolves this checkout.
# Its parent also contains an unrelated x2camf source tree which must not
# shadow the installed X2CAMF extension.
unset PYTHONPATH
python_bin=/home/Yxwxwx/code/socutils/.venv/bin/python
test_directory=tests/nevpt2_mps_response
result_directory="$test_directory/f_sa6_ground_run"
mkdir -p "$result_directory"
printf 'Host=%s PID=%s CPU_threads=%s scratch=%s\n' "$(hostname)" "$$" "$OMP_NUM_THREADS" "$task_scratch"
for stage in mcscf hybrid full compare; do
    printf 'Stage=%s started=%s\n' "$stage" "$(date -Is)"
    stage_args=(--directory "$result_directory")
    if [[ "$stage" == hybrid ]]; then
        # Energy comparison does not need the exponential validation-only
        # coefficient expansion. The production solver and RDM guard remain.
        stage_args+=(--skip-residual-audit)
    fi
    if [[ "$stage" == hybrid && -n "${NEVPT2_RESPONSE_OPTIONS:-}" ]]; then
        stage_args+=(--response-options "$NEVPT2_RESPONSE_OPTIONS")
    fi
    /usr/bin/time -v "$python_bin" -u -m tests.nevpt2_mps_response.f_atom "$stage" \
        "${stage_args[@]}" \
        > "$result_directory/${stage}.out" 2> "$result_directory/${stage}.time"
done
