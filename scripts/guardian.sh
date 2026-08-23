#!/usr/bin/env bash
# Resubmit a SLURM job until its checkpoint is complete.
#
# Exists because Mahamathi's a100 partition caps walltime at 24h and jobs on cn6
# have a history of being killed early ("CANCELLED by 0", see docs/HANDOFF.md
# section 9). Both pipelines checkpoint per question and resume, so the fix is
# simply to keep resubmitting until the row count is reached.
#
#   bash scripts/guardian.sh                      # main OSAM pipeline (1540 rows)
#   bash scripts/guardian.sh ablation             # IterRet-only ablation (1540 rows)
#   bash scripts/guardian.sh <slurm> <out> <n>    # anything else
#
# Pass extra sbatch flags via GUARDIAN_SBATCH_ARGS, e.g.
#   GUARDIAN_SBATCH_ARGS="--partition=a100" bash scripts/guardian.sh ablation
#
# Run under tmux -- it polls until done.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

# 1540 = 1986 LoCoMo questions - 446 adversarial. BOTH pipelines now exclude
# adversarial, so neither output file can ever exceed 1540 lines. Do not
# "correct" this to 1986 -- that would resubmit forever on a finished run.
TARGET_ROWS_DEFAULT=1540

case "${1:-main}" in
  main)
    SLURM_SCRIPT="${HERE}/run_full_pipeline.slurm"
    OUTPUT_FILE="${CAIMMS_OUTPUT_DIR}/workmem_iterret_full.jsonl"
    TARGET_ROWS="${TARGET_ROWS_DEFAULT}"
    ;;
  ablation)
    SLURM_SCRIPT="${HERE}/run_ablation.slurm"
    OUTPUT_FILE="${CAIMMS_OUTPUT_DIR}/workmem_ablation_direct.jsonl"
    TARGET_ROWS="${TARGET_ROWS_DEFAULT}"
    ;;
  *)
    SLURM_SCRIPT="$1"
    OUTPUT_FILE="${2:?need an output file as arg 2}"
    TARGET_ROWS="${3:-${TARGET_ROWS_DEFAULT}}"
    ;;
esac

[ -f "${SLURM_SCRIPT}" ] || { echo "[guardian] no such slurm script: ${SLURM_SCRIPT}"; exit 1; }

echo "[guardian] script : ${SLURM_SCRIPT}"
echo "[guardian] output : ${OUTPUT_FILE}"
echo "[guardian] target : ${TARGET_ROWS} rows"
echo "[guardian] sbatch : ${GUARDIAN_SBATCH_ARGS:-<none>}"

while true; do
    N_DONE=$([ -f "${OUTPUT_FILE}" ] && wc -l < "${OUTPUT_FILE}" || echo 0)
    if [ "${N_DONE}" -ge "${TARGET_ROWS}" ]; then
        echo "[guardian] Complete (${N_DONE}/${TARGET_ROWS}). Stopping."
        exit 0
    fi
    echo "[guardian] Submitting (${N_DONE}/${TARGET_ROWS} done)..."
    # shellcheck disable=SC2086
    JOB_ID=$(cd "${CAIMMS_ROOT}" && sbatch --parsable ${GUARDIAN_SBATCH_ARGS:-} "${SLURM_SCRIPT}")
    if [ -z "${JOB_ID}" ]; then
        echo "[guardian] sbatch failed. Retrying in 30s..."; sleep 30; continue
    fi
    echo "[guardian] Job ${JOB_ID} submitted at $(date)."
    while squeue -j "${JOB_ID}" 2>/dev/null | grep -q "${JOB_ID}"; do sleep 30; done
    N_AFTER=$([ -f "${OUTPUT_FILE}" ] && wc -l < "${OUTPUT_FILE}" || echo 0)
    echo "[guardian] Job ${JOB_ID} ended at $(date). ${N_AFTER}/${TARGET_ROWS} done."
    # Guard against a job that dies instantly and makes no progress -- without
    # this the loop would hammer the scheduler in a tight submit/fail cycle.
    if [ "${N_AFTER}" -le "${N_DONE}" ]; then
        echo "[guardian] WARNING: no progress this job. Backing off 120s."
        echo "[guardian] Check the job log before letting this continue."
        sleep 120
    else
        sleep 5
    fi
done
